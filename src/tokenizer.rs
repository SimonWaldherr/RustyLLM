// tokenizer.rs — BPE tokenizer from GGUF metadata
//
// Supports SentencePiece-style BPE with merge scores, byte fallback,
// and special token handling (BOS/EOS).

use crate::gguf::MetaValue;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};

#[derive(Clone, Copy, PartialEq, Eq)]
enum TokenizerMode {
    SentencePiece,
    Gpt2Bpe,
    Gemma4Bpe,
    /// BERT-style WordPiece (nomic-bert, bge, MiniLM …): lowercase +
    /// accent-strip normalization, greedy longest-match, `[CLS]`/`[SEP]`
    /// framing. GGUF stores the vocab in phantom-space form (`##cont` → `cont`,
    /// word-start → `\u{2581}tok`), so matching uses a `\u{2581}` prefix.
    WordPiece,
}

pub struct Tokenizer {
    vocab: Vec<String>,
    scores: Vec<f32>,
    token_to_id: HashMap<String, u32>,
    bpe_merges: HashMap<(u32, u32), (usize, u32)>,
    /// Text-keyed merges for Gemma 4, nested left-then-right so a pair can be
    /// looked up from two `&str` without building an owned key for each probe.
    bpe_text_merges: HashMap<String, HashMap<String, (usize, u32)>>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    byte_token_ids: [Option<u32>; 256],
    mode: TokenizerMode,
    /// Mistral's Tekken pre-tokenizer: same byte-level vocabulary as GPT-2, but
    /// a different split and a whole-word vocabulary shortcut before merges.
    tekken: bool,
    add_bos_token: bool,
    /// Longest vocab entry in chars; bounds the WordPiece greedy match window.
    max_wp_token_chars: usize,
    pub bos_id: u32,
    pub eos_id: u32,
    /// WordPiece unknown-token id (`tokenizer.ggml.unknown_token_id`).
    pub unk_id: u32,
    /// WordPiece separator id (`tokenizer.ggml.seperator_token_id`, note the
    /// llama.cpp misspelling), appended after the encoded pieces.
    pub sep_id: u32,
}

impl Tokenizer {
    /// Builds a tokenizer from GGUF tokenizer metadata.
    pub fn from_metadata(metadata: &HashMap<String, MetaValue>) -> Self {
        let vocab = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_string_array())
            .expect("Missing tokenizer.ggml.tokens");

        let scores = metadata
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.as_f32_array())
            .unwrap_or_else(|| vec![0.0; vocab.len()]);

        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (i, tok) in vocab.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as u32);
        }

        let mut bpe_merges = HashMap::new();
        let mut bpe_text_merges: HashMap<String, HashMap<String, (usize, u32)>> = HashMap::new();
        if let Some(merges) = metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_string_array())
        {
            for (rank, merge) in merges.iter().enumerate() {
                if let Some((left, right)) = merge.split_once(' ') {
                    let mut merged = String::with_capacity(left.len() + right.len());
                    merged.push_str(left);
                    merged.push_str(right);
                    if let Some(&merged_id) = token_to_id.get(&merged) {
                        bpe_text_merges
                            .entry(left.to_string())
                            .or_default()
                            .insert(right.to_string(), (rank, merged_id));
                        if let (Some(&left_id), Some(&right_id)) =
                            (token_to_id.get(left), token_to_id.get(right))
                        {
                            bpe_merges.insert((left_id, right_id), (rank, merged_id));
                        }
                    }
                }
            }
        }

        let model = metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let pre = metadata
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // GGUF tokenizer metadata is not fully standardized across model
        // families, so mode selection uses a few conservative hints.
        let pre_lower = pre.to_ascii_lowercase();
        // Tekken shares GPT-2's byte-level vocabulary but splits differently,
        // so it is a distinct pre-tokenizer rather than a distinct mode.
        let tekken = pre_lower == "tekken";
        let mode = if model.eq_ignore_ascii_case("gemma4") {
            TokenizerMode::Gemma4Bpe
        } else if model.eq_ignore_ascii_case("bert") {
            TokenizerMode::WordPiece
        } else if model.eq_ignore_ascii_case("gpt2")
            || tekken
            || pre_lower.contains("qwen")
            || pre_lower.contains("gpt")
        {
            TokenizerMode::Gpt2Bpe
        } else {
            TokenizerMode::SentencePiece
        };

        let (byte_encoder, byte_decoder) = build_byte_maps();
        let mut byte_token_ids = [None; 256];
        for byte in 0u16..=255 {
            let byte_tok = format!("<0x{:02X}>", byte);
            byte_token_ids[byte as usize] = token_to_id.get(&byte_tok).copied();
        }

        let bos_id = metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(1);
        let eos_id = metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(2);

        let add_bos_token = metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| match v {
                MetaValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(true);

        // WordPiece special ids: BERT GGUFs default CLS=101 (bos), UNK=100,
        // SEP=102 (also the misspelled `seperator` key), falling back to eos.
        let unk_id = metadata
            .get("tokenizer.ggml.unknown_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(100);
        let sep_id = metadata
            .get("tokenizer.ggml.seperator_token_id")
            .or_else(|| metadata.get("tokenizer.ggml.separator_token_id"))
            .and_then(|v| v.as_u32())
            .unwrap_or(eos_id);

        let max_wp_token_chars = if mode == TokenizerMode::WordPiece {
            vocab.iter().map(|t| t.chars().count()).max().unwrap_or(1)
        } else {
            0
        };

        Self {
            vocab,
            scores,
            token_to_id,
            bpe_merges,
            bpe_text_merges,
            byte_encoder,
            byte_decoder,
            byte_token_ids,
            mode,
            tekken,
            add_bos_token,
            max_wp_token_chars,
            bos_id,
            eos_id,
            unk_id,
            sep_id,
        }
    }

    /// BPE encode: start with character/byte tokens, then greedily merge.
    ///
    /// For WordPiece (encoder models) this frames the sequence as
    /// `[CLS] … [SEP]`, matching llama.cpp's BERT tokenizer.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        if self.add_bos_token {
            tokens.push(self.bos_id);
        }
        tokens.extend_from_slice(&self.encode_without_bos(text));
        if self.mode == TokenizerMode::WordPiece {
            tokens.push(self.sep_id);
        }
        tokens
    }

    /// Encodes text into token IDs without adding BOS.
    pub fn encode_without_bos(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();

        if text.is_empty() {
            return tokens;
        }

        let encoded = match self.mode {
            TokenizerMode::SentencePiece => self.encode_sentencepiece(text),
            TokenizerMode::Gpt2Bpe => self.encode_gpt2_bpe(text),
            TokenizerMode::Gemma4Bpe => self.encode_gemma4_bpe(text),
            TokenizerMode::WordPiece => self.encode_wordpiece(text),
        };
        tokens.extend_from_slice(&encoded);
        tokens
    }

    /// Raw decode (internal representation with ▁)
    fn decode_raw(&self, id: u32) -> &str {
        if (id as usize) < self.vocab.len() {
            &self.vocab[id as usize]
        } else {
            ""
        }
    }

    /// User-facing decode: ▁ → space, handle byte tokens.
    ///
    /// A token may end mid-codepoint, in which case this yields U+FFFD. Callers
    /// that emit tokens one at a time should decode through [`Utf8Stitcher`]
    /// instead so split characters survive.
    pub fn decode_token(&self, id: u32) -> String {
        String::from_utf8_lossy(&self.decode_token_bytes(id)).into_owned()
    }

    /// Decodes one token to its raw bytes, before any UTF-8 validation.
    ///
    /// Byte-level BPE stores text as remapped bytes and SentencePiece falls
    /// back to `<0xHH>` tokens, so a single character can span several tokens.
    /// Returning bytes lets a caller reassemble those characters; converting to
    /// `String` per token cannot, because the fragments are not valid UTF-8 on
    /// their own.
    pub fn decode_token_bytes(&self, id: u32) -> Vec<u8> {
        let raw = self.decode_raw(id);

        if self.mode == TokenizerMode::Gpt2Bpe {
            let mut bytes = Vec::with_capacity(raw.len());
            for ch in raw.chars() {
                match self.byte_decoder.get(&ch) {
                    Some(&b) => bytes.push(b),
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
            return bytes;
        }

        // SentencePiece byte fallback: `<0xHH>` is one raw byte, not the
        // Latin-1 character with that code point.
        if raw.starts_with("<0x")
            && raw.ends_with('>')
            && raw.len() == 6
            && let Ok(byte) = u8::from_str_radix(&raw[3..5], 16)
        {
            return vec![byte];
        }

        raw.replace('\u{2581}', " ").into_bytes()
    }

    /// Returns the number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Returns the raw tokenizer vocabulary entry for a token ID.
    pub fn raw_token(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(String::as_str)
    }

    /// Returns the tokenizer score for a token ID when GGUF metadata provided one.
    pub fn token_score(&self, id: u32) -> Option<f32> {
        self.scores.get(id as usize).copied()
    }

    /// Returns whether normal text encoding prepends the BOS token.
    pub fn adds_bos_token(&self) -> bool {
        self.add_bos_token
    }

    /// Looks up the ID of a special token string.
    pub fn special_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Encodes text with the SentencePiece-style BPE path.
    fn encode_sentencepiece(&self, text: &str) -> Vec<u32> {
        // SentencePiece models encode word starts with U+2581, so we inject a
        // leading space before splitting to preserve first-token behavior.
        let mut current_tokens = Vec::with_capacity(text.len() + 1);
        self.encode_piece("\u{2581}", &mut current_tokens);
        for ch in text.chars() {
            if ch == ' ' {
                self.encode_piece("\u{2581}", &mut current_tokens);
            } else {
                let mut buf = [0u8; 4];
                self.encode_piece(ch.encode_utf8(&mut buf), &mut current_tokens);
            }
        }

        // Iterative BPE merge: repeatedly take the highest-scoring adjacent
        // pair. Unlike the byte-level paths there is no pre-tokenizer here, so
        // this list is the whole prompt and the merge order has to come from a
        // heap rather than a rescan to stay tractable.
        let mut merged = String::new();
        merge_symbols(&mut current_tokens, |&left, &right| {
            merged.clear();
            merged.push_str(self.decode_raw(left));
            merged.push_str(self.decode_raw(right));
            let id = *self.token_to_id.get(merged.as_str())?;
            let score = self.scores.get(id as usize).copied().unwrap_or(0.0);
            // Rejecting `-inf` and NaN mirrors the strict improvement the
            // previous scan required, so scoring behaviour is unchanged.
            (score > f32::NEG_INFINITY).then_some((ScoreKey(score), id))
        });
        current_tokens
    }

    /// Encodes text with byte-level GPT-2 BPE.
    fn encode_gpt2_bpe(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let pieces = if self.tekken {
            pretokenize_tekken(text)
        } else {
            pretokenize_gpt2(text)
        };
        // Reused across pre-tokens so the whole-word lookup below does not
        // allocate once per piece.
        let mut encoded = String::new();
        let mut symbols: Vec<u32> = Vec::new();
        for piece in pieces {
            // GPT-2 style BPE operates on a reversible byte-level alphabet
            // before merge ranks are applied.
            //
            // Tekken sets `ignore_merges`: if the byte-encoded pre-token exists
            // verbatim in the vocabulary it is emitted as one token and the
            // merge loop is skipped entirely, which can differ from what the
            // merges alone would produce.
            if self.tekken {
                encoded.clear();
                encoded.extend(
                    piece
                        .as_bytes()
                        .iter()
                        .map(|&byte| self.byte_encoder[byte as usize]),
                );
                if let Some(&id) = self.token_to_id.get(encoded.as_str()) {
                    out.push(id);
                    continue;
                }
            }

            symbols.clear();
            symbols.reserve(piece.len());
            for &byte in piece.as_bytes() {
                let ch = self.byte_encoder[byte as usize];
                let mut buf = [0u8; 4];
                let symbol = ch.encode_utf8(&mut buf);
                if let Some(&id) = self.token_to_id.get(symbol) {
                    symbols.push(id);
                } else {
                    self.encode_piece(symbol, &mut symbols);
                }
            }

            // Lower rank wins, so the key is reversed to keep the heap's
            // largest-first order.
            merge_symbols(&mut symbols, |&left, &right| {
                self.bpe_merges
                    .get(&(left, right))
                    .map(|&(rank, merged_id)| (Reverse(rank), merged_id))
            });

            out.extend(symbols.iter().copied());
        }
        out
    }

    /// Encodes text with Gemma 4's SPM-style BPE over raw UTF-8.
    fn encode_gemma4_bpe(&self, text: &str) -> Vec<u32> {
        let normalized = text.replace(' ', "\u{2581}");
        let mut out = Vec::new();

        for piece in split_gemma4_pieces(&normalized) {
            if piece.is_empty() {
                continue;
            }

            let is_newlines = piece.as_bytes().iter().all(|&b| b == b'\n');
            if is_newlines {
                if let Some(&id) = self.token_to_id.get(piece) {
                    out.push(id);
                    continue;
                }
            }

            // Symbols stay as borrowed slices: the initial ones point into the
            // piece and every merged one into the vocabulary, so a merge copies
            // no text at all.
            let mut symbols: Vec<&str> = piece
                .char_indices()
                .map(|(at, ch)| &piece[at..at + ch.len_utf8()])
                .collect();

            merge_symbols(&mut symbols, |left, right| {
                self.bpe_text_merges
                    .get(*left)?
                    .get(*right)
                    .map(|&(rank, merged_id)| (Reverse(rank), self.decode_raw(merged_id)))
            });

            for symbol in symbols {
                self.encode_piece(symbol, &mut out);
            }
        }

        out
    }

    /// Encodes text with BERT-style WordPiece.
    ///
    /// Mirrors llama.cpp's WPM tokenizer: normalize (lowercase, strip accents,
    /// drop control chars), split on whitespace, isolate punctuation / ASCII
    /// symbols / CJK as single-char words, then greedily match the longest
    /// vocab piece at each position. The GGUF vocab is phantom-space form, so
    /// each word is prefixed with `\u{2581}` and word-start pieces are matched
    /// with that prefix while continuation pieces are matched bare. Any word
    /// with a position that fails to match becomes a single `[UNK]`.
    fn encode_wordpiece(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for word in wordpiece_split(text) {
            self.encode_wordpiece_word(&word, &mut out);
        }
        out
    }

    /// Greedily segments one normalized word into WordPiece token ids.
    fn encode_wordpiece_word(&self, word: &str, out: &mut Vec<u32>) {
        if word.is_empty() {
            return;
        }
        // Phantom-space form: a leading `\u{2581}` marks the word start, and
        // continuation pieces are stored without the `##` prefix.
        let prefixed = format!("\u{2581}{word}");
        // Char-start byte offsets let every candidate window be a borrowed
        // slice; collecting the chars instead would rebuild a `String` for each
        // of the (up to `max_wp_token_chars`) windows tried per position.
        let mut offsets: Vec<usize> = prefixed.char_indices().map(|(at, _)| at).collect();
        offsets.push(prefixed.len());
        let n = offsets.len() - 1;

        // Pieces go straight into `out`; a word that fails to match rewinds to
        // this mark instead of accumulating in a throwaway vector.
        let mark = out.len();
        let mut start = 0usize;
        while start < n {
            let mut matched: Option<u32> = None;
            let mut matched_end = start;
            // Longest-match: try the widest window first.
            let mut end = (start + self.max_wp_token_chars).min(n);
            while end > start {
                if let Some(&id) = self
                    .token_to_id
                    .get(&prefixed[offsets[start]..offsets[end]])
                {
                    matched = Some(id);
                    matched_end = end;
                    break;
                }
                end -= 1;
            }
            match matched {
                Some(id) => {
                    out.push(id);
                    start = matched_end;
                }
                None => {
                    // Unmatched position ⇒ the whole word is unknown.
                    out.truncate(mark);
                    out.push(self.unk_id);
                    return;
                }
            }
        }
    }

    /// Maps token pieces to IDs, falling back to byte tokens when needed.
    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        if let Some(&id) = self.token_to_id.get(piece) {
            out.push(id);
        } else {
            for &byte in piece.as_bytes() {
                if let Some(id) = self.byte_token_ids[byte as usize] {
                    out.push(id);
                }
            }
        }
    }
}

/// A queued merge of two adjacent symbols, ordered so the best pops first.
struct MergeCandidate<K, S> {
    key: K,
    /// The symbol that replaces the pair, computed when the pair was queued.
    merged: S,
    /// Slot of the left symbol. Also the tie-break, so equal keys resolve to
    /// the leftmost pair exactly as a left-to-right scan would.
    left: u32,
    right: u32,
    /// Slot versions at queue time; a later merge bumps them, which is how a
    /// candidate that has since become invalid is recognised.
    left_version: u32,
    right_version: u32,
}

impl<K: Ord, S> Ord for MergeCandidate<K, S> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key
            .cmp(&other.key)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl<K: Ord, S> PartialOrd for MergeCandidate<K, S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord, S> PartialEq for MergeCandidate<K, S> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<K: Ord, S> Eq for MergeCandidate<K, S> {}

/// Ordering key for a merge table scored with floats.
///
/// Only finite scores are ever queued — the scan this replaces started from
/// negative infinity and demanded a strict improvement, so `-inf` and NaN were
/// unreachable — which makes `total_cmp` a genuine total order here.
struct ScoreKey(f32);

impl Ord for ScoreKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for ScoreKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ScoreKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}

impl Eq for ScoreKey {}

/// Greedily merges adjacent symbols until no pair in the table is left.
///
/// The textbook loop rescans every adjacent pair after each merge and deletes
/// with `Vec::remove`, so it costs O(n²). That is invisible for one short
/// pre-token, but SentencePiece has no pre-tokenizer — its symbol list is the
/// *entire* prompt — so the cost lands squarely on prompt processing. Holding
/// candidates in a heap and the symbols in a doubly linked list makes the pass
/// O(n log n): a merge retires one entry and queues at most two more, and
/// unlinking is O(1).
///
/// `pair` returns the ordering key and merged symbol for an adjacent pair, or
/// `None` when the pair cannot merge. Larger keys win and equal keys resolve
/// leftmost, so the result is identical to the scan this replaces.
fn merge_symbols<S, K, F>(symbols: &mut Vec<S>, mut pair: F)
where
    K: Ord,
    F: FnMut(&S, &S) -> Option<(K, S)>,
{
    let count = symbols.len();
    if count < 2 {
        return;
    }

    // Slots keep their original index for the whole pass; only the links move.
    let mut prev: Vec<i32> = (0..count as i32).map(|slot| slot - 1).collect();
    let mut next: Vec<i32> = (0..count as i32)
        .map(|slot| {
            if slot + 1 < count as i32 {
                slot + 1
            } else {
                -1
            }
        })
        .collect();
    let mut alive = vec![true; count];
    let mut version = vec![0u32; count];

    let mut heap: BinaryHeap<MergeCandidate<K, S>> = BinaryHeap::new();
    let queue = |heap: &mut BinaryHeap<MergeCandidate<K, S>>,
                 symbols: &[S],
                 version: &[u32],
                 left: usize,
                 right: usize,
                 pair: &mut F| {
        if let Some((key, merged)) = pair(&symbols[left], &symbols[right]) {
            heap.push(MergeCandidate {
                key,
                merged,
                left: left as u32,
                right: right as u32,
                left_version: version[left],
                right_version: version[right],
            });
        }
    };

    for left in 0..count - 1 {
        queue(&mut heap, symbols, &version, left, left + 1, &mut pair);
    }

    while let Some(candidate) = heap.pop() {
        let left = candidate.left as usize;
        let right = candidate.right as usize;
        // A merge elsewhere may have consumed either side or broken the
        // adjacency this candidate was queued for. Both `alive` checks are
        // load-bearing: a slot consumed as a right side keeps its version and
        // its stale `next` link, so without the left check it would still look
        // mergeable to a candidate queued before it died.
        if !alive[left]
            || !alive[right]
            || next[left] != right as i32
            || version[left] != candidate.left_version
            || version[right] != candidate.right_version
        {
            continue;
        }

        symbols[left] = candidate.merged;
        version[left] += 1;
        // Only the right side is ever unlinked, so slot 0 always survives and
        // the surviving slots stay in ascending order.
        alive[right] = false;

        let after = next[right];
        next[left] = after;
        if after >= 0 {
            prev[after as usize] = left as i32;
        }

        // The rewritten symbol invalidates both of its pairings, so requeue
        // whichever neighbours it still has.
        let before = prev[left];
        if before >= 0 {
            queue(
                &mut heap,
                symbols,
                &version,
                before as usize,
                left,
                &mut pair,
            );
        }
        if after >= 0 {
            queue(
                &mut heap,
                symbols,
                &version,
                left,
                after as usize,
                &mut pair,
            );
        }
    }

    let mut slot = 0usize;
    symbols.retain(|_| {
        let keep = alive[slot];
        slot += 1;
        keep
    });
}

/// Reassembles UTF-8 text from a stream of separately decoded tokens.
///
/// Byte-level BPE routinely splits one code point across two tokens — every
/// emoji and most CJK text — so converting each token independently emits
/// U+FFFD for the halves. This buffers a trailing sequence that is still
/// completable and releases it once the next token finishes the character.
#[derive(Default)]
pub struct Utf8Stitcher {
    pending: Vec<u8>,
}

impl Utf8Stitcher {
    /// Creates an empty stitcher.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds one token's bytes and returns the text that is now complete.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    out.push_str(text);
                    self.pending.clear();
                    return out;
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    // SAFETY-equivalent: `valid_up_to` is a validated boundary.
                    out.push_str(&String::from_utf8_lossy(&self.pending[..valid]));
                    match error.error_len() {
                        // Truncated but still possible — wait for more bytes.
                        None => {
                            self.pending.drain(..valid);
                            return out;
                        }
                        // Genuinely invalid: report it and skip past it.
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..valid + bad);
                        }
                    }
                }
            }
        }
    }

    /// Releases any trailing incomplete sequence at end of generation.
    ///
    /// A truncated character cannot be completed once generation stops, so it
    /// surfaces as U+FFFD rather than being silently dropped.
    pub fn flush(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }

    /// Reports whether bytes are buffered awaiting completion.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }
}

/// Splits text into Mistral Tekken BPE pre-token chunks.
///
/// Mirrors llama.cpp's TEKKEN regex, including its quirk: the upstream pattern
/// uses `\p{Lu}`/`\p{Ll}`, but llama.cpp collapses every non-ASCII letter into a
/// single class, so case boundaries only exist for `a-z`/`A-Z`. Reproducing that
/// keeps token counts identical to the reference rather than "more correct" and
/// therefore different.
///
/// Three differences from the GPT-2 splitter change token counts materially:
/// each digit becomes its own pre-token, at most one leading character attaches
/// to a word, and a whitespace run ending in a newline is isolated.
fn pretokenize_tekken(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let count = chars.len();
    // Byte offset of char `k`, or the end of the string.
    let offset = |k: usize| chars.get(k).map_or(text.len(), |(at, _)| *at);

    let is_letter = |c: char| c.is_alphabetic();
    let is_digit = |c: char| c.is_numeric();
    // Collapsed classes: a non-ASCII letter satisfies both, exactly as in the
    // reference implementation.
    let is_upper = |c: char| is_letter(c) && !c.is_ascii_lowercase();
    let is_lower = |c: char| is_letter(c) && !c.is_ascii_uppercase();
    let is_lead = |c: char| c != '\r' && c != '\n' && !is_letter(c) && !is_digit(c);
    let is_newline = |c: char| c == '\r' || c == '\n';

    let mut pieces = Vec::new();
    let mut i = 0usize;
    while i < count {
        let mut end: Option<usize> = None;

        // Branches 1 and 2: an optional single lead character, then a run of
        // "uppercase" letters and a run of "lowercase" ones.
        for take_lead in [true, false] {
            let mut k = i;
            if take_lead {
                if !is_lead(chars[k].1) {
                    continue;
                }
                k += 1;
            }
            let upper_start = k;
            while k < count && is_upper(chars[k].1) {
                k += 1;
            }
            let upper_end = k;

            // Branch 1 needs at least one trailing lowercase letter; give back
            // uppercase characters one at a time, as regex backtracking would.
            let mut give_back = upper_end;
            loop {
                let mut j = give_back;
                while j < count && is_lower(chars[j].1) {
                    j += 1;
                }
                if j > give_back {
                    end = Some(j);
                    break;
                }
                if give_back == upper_start {
                    break;
                }
                give_back -= 1;
            }
            if end.is_some() {
                break;
            }

            // Branch 2: at least one uppercase, trailing lowercase optional.
            if upper_end > upper_start {
                let mut j = upper_end;
                while j < count && is_lower(chars[j].1) {
                    j += 1;
                }
                end = Some(j);
                break;
            }
        }

        // Branch 3: exactly one digit — never a run.
        if end.is_none() && is_digit(chars[i].1) {
            end = Some(i + 1);
        }

        // Branch 4: optional single space, punctuation run, trailing newlines
        // or slashes.
        if end.is_none() {
            let mut k = i;
            if chars[k].1 == ' ' {
                k += 1;
            }
            let symbol_start = k;
            while k < count {
                let c = chars[k].1;
                if c.is_whitespace() || is_letter(c) || is_digit(c) {
                    break;
                }
                k += 1;
            }
            if k > symbol_start {
                while k < count && (is_newline(chars[k].1) || chars[k].1 == '/') {
                    k += 1;
                }
                end = Some(k);
            }
        }

        // Branches 5 to 7, all whitespace.
        if end.is_none() && chars[i].1.is_whitespace() {
            let mut run = i;
            while run < count && chars[run].1.is_whitespace() {
                run += 1;
            }
            // Branch 5: a run ending in newlines is one pre-token, cut after
            // the last newline group inside it.
            let last_newline = (i..run).rev().find(|&k| is_newline(chars[k].1));
            if let Some(last) = last_newline {
                let mut group_end = last;
                while group_end < run && is_newline(chars[group_end].1) {
                    group_end += 1;
                }
                end = Some(group_end);
            } else if run == count {
                // Branch 6: trailing whitespace at end of input.
                end = Some(run);
            } else {
                // Branch 6 backtracks so the final space attaches to the next
                // word as its lead character; branch 7 covers a lone space.
                end = Some(if run - i > 1 { run - 1 } else { run });
            }
        }

        let stop = end.unwrap_or(i + 1);
        pieces.push(&text[offset(i)..offset(stop)]);
        i = stop;
    }

    pieces
}

/// Splits text into GPT-2 BPE pre-token chunks.
fn pretokenize_gpt2(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut i = 0usize;

    // This is a lightweight approximation of GPT-2's regex pre-tokenizer:
    // group leading whitespace with the following token and split runs of
    // letters, digits, and punctuation separately.
    while i < text.len() {
        let start = i;
        let mut had_space = false;
        while i < text.len() {
            let ch = text[i..]
                .chars()
                .next()
                .expect("byte index is on a char boundary");
            if !ch.is_whitespace() {
                break;
            }
            had_space = true;
            i += ch.len_utf8();
        }

        if i >= text.len() {
            if had_space {
                pieces.push(&text[start..i]);
            }
            break;
        }

        let token_start = i;
        let c = text[i..]
            .chars()
            .next()
            .expect("byte index is on a char boundary");
        if c.is_alphabetic() {
            while i < text.len() {
                let ch = text[i..]
                    .chars()
                    .next()
                    .expect("byte index is on a char boundary");
                if !ch.is_alphabetic() {
                    break;
                }
                i += ch.len_utf8();
            }
        } else if c.is_numeric() {
            while i < text.len() {
                let ch = text[i..]
                    .chars()
                    .next()
                    .expect("byte index is on a char boundary");
                if !ch.is_numeric() {
                    break;
                }
                i += ch.len_utf8();
            }
        } else {
            while i < text.len() {
                let ch = text[i..]
                    .chars()
                    .next()
                    .expect("byte index is on a char boundary");
                if ch.is_whitespace() || ch.is_alphabetic() || ch.is_numeric() {
                    break;
                }
                i += ch.len_utf8();
            }
        }

        let piece_start = if had_space { start } else { token_start };
        pieces.push(&text[piece_start..i]);
    }

    pieces
}

/// Splits Gemma 4 text into non-newline and newline runs.
fn split_gemma4_pieces(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut start = 0usize;
    let mut last_is_newline: Option<bool> = None;

    for (idx, ch) in text.char_indices() {
        let is_newline = ch == '\n';
        if let Some(prev) = last_is_newline {
            if prev != is_newline {
                pieces.push(&text[start..idx]);
                start = idx;
            }
        }
        last_is_newline = Some(is_newline);
    }

    if start < text.len() {
        pieces.push(&text[start..]);
    }
    pieces
}

/// Splits text into normalized WordPiece words.
///
/// Applies llama.cpp's WPM preprocessing: lowercase, strip accents, drop
/// control / replacement characters, split on Unicode whitespace, and emit
/// each punctuation char, ASCII symbol, or CJK char as its own word.
fn wordpiece_split(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let flush = |current: &mut String, words: &mut Vec<String>| {
        if !current.is_empty() {
            words.push(std::mem::take(current));
        }
    };

    for ch in text.chars() {
        // Drop the null and replacement characters and C0/C1 control chars.
        if ch == '\u{0}' || ch == '\u{fffd}' || ch.is_control() {
            continue;
        }
        if ch.is_whitespace() {
            flush(&mut current, &mut words);
            continue;
        }
        if is_wordpiece_standalone(ch) {
            flush(&mut current, &mut words);
            words.push(strip_accents_lower(ch));
            continue;
        }
        current.push_str(&strip_accents_lower(ch));
    }
    flush(&mut current, &mut words);
    words
}

/// Reports whether a character is tokenized as its own WordPiece word
/// (punctuation, ASCII symbols, or CJK ideographs).
fn is_wordpiece_standalone(ch: char) -> bool {
    if ch.is_ascii_punctuation() {
        return true;
    }
    let cp = ch as u32;
    // ASCII symbols that are not is_ascii_punctuation (none extra) plus the
    // Unicode CJK ideograph ranges llama.cpp isolates.
    matches!(cp,
        0x3000..=0x303F   // CJK symbols and punctuation
        | 0x4E00..=0x9FFF // CJK unified ideographs
        | 0x3400..=0x4DBF // extension A
        | 0x20000..=0x2A6DF
        | 0x2A700..=0x2B73F
        | 0xF900..=0xFAFF // compatibility ideographs
        | 0xFF00..=0xFFEF // halfwidth/fullwidth forms
    ) || (!ch.is_alphanumeric()
        && !ch.is_whitespace()
        && cp > 0x7F
        && is_unicode_punct_or_symbol(ch))
}

/// Approximates Unicode punctuation/symbol categories for the standalone check.
fn is_unicode_punct_or_symbol(ch: char) -> bool {
    let cp = ch as u32;
    matches!(cp,
        0x2000..=0x206F  // general punctuation
        | 0x2070..=0x209F
        | 0x20A0..=0x20CF // currency symbols
        | 0x2100..=0x214F // letterlike symbols
        | 0x2190..=0x2BFF // arrows, math, misc symbols
    )
}

/// Lowercases a character and strips diacritics for the Latin ranges (the
/// common case for nomic-embed inputs). Combining marks are dropped; other
/// scripts pass through lowercased. Full Unicode NFD parity is out of scope —
/// rare non-Latin accents may diverge from llama.cpp.
fn strip_accents_lower(ch: char) -> String {
    // Drop combining diacritical marks outright.
    if ('\u{0300}'..='\u{036F}').contains(&ch) {
        return String::new();
    }
    let base = latin_deaccent(ch).unwrap_or(ch);
    base.to_lowercase().collect()
}

/// Maps accented Latin-1 Supplement / Latin Extended-A letters to their base
/// letter. Returns `None` when there is no decomposition.
fn latin_deaccent(ch: char) -> Option<char> {
    let base = match ch {
        'À'..='Å' | 'à'..='å' | 'Ā' | 'ā' | 'Ă' | 'ă' | 'Ą' | 'ą' => {
            if ch.is_uppercase() { 'A' } else { 'a' }
        }
        'Ç' | 'ç' | 'Ć' | 'ć' | 'Ĉ' | 'ĉ' | 'Ċ' | 'ċ' | 'Č' | 'č' => {
            if ch.is_uppercase() { 'C' } else { 'c' }
        }
        'È'..='Ë' | 'è'..='ë' | 'Ē' | 'ē' | 'Ĕ' | 'ĕ' | 'Ė' | 'ė' | 'Ę' | 'ę' | 'Ě' | 'ě' => {
            if ch.is_uppercase() {
                'E'
            } else {
                'e'
            }
        }
        'Ì'..='Ï' | 'ì'..='ï' | 'Ĩ' | 'ĩ' | 'Ī' | 'ī' | 'Ĭ' | 'ĭ' | 'Į' | 'į' | 'İ' => {
            if ch.is_uppercase() {
                'I'
            } else {
                'i'
            }
        }
        'Ñ' | 'ñ' | 'Ń' | 'ń' | 'Ņ' | 'ņ' | 'Ň' | 'ň' => {
            if ch.is_uppercase() {
                'N'
            } else {
                'n'
            }
        }
        'Ò'..='Ö' | 'ò'..='ö' | 'Ō' | 'ō' | 'Ŏ' | 'ŏ' | 'Ő' | 'ő' | 'Ø' | 'ø' => {
            if ch.is_uppercase() { 'O' } else { 'o' }
        }
        'Ù'..='Ü' | 'ù'..='ü' | 'Ũ' | 'ũ' | 'Ū' | 'ū' | 'Ŭ' | 'ŭ' | 'Ů' | 'ů' | 'Ű' | 'ű' => {
            if ch.is_uppercase() {
                'U'
            } else {
                'u'
            }
        }
        'Ý' | 'ý' | 'ÿ' | 'Ŷ' | 'ŷ' | 'Ÿ' => {
            if ch.is_uppercase() {
                'Y'
            } else {
                'y'
            }
        }
        'Š' | 'š' | 'Ś' | 'ś' | 'Ŝ' | 'ŝ' | 'Ş' | 'ş' => {
            if ch.is_uppercase() {
                'S'
            } else {
                's'
            }
        }
        'Ž' | 'ž' | 'Ź' | 'ź' | 'Ż' | 'ż' => {
            if ch.is_uppercase() {
                'Z'
            } else {
                'z'
            }
        }
        _ => return None,
    };
    Some(base)
}

/// Builds reversible GPT-2 byte encoder and decoder tables.
fn build_byte_maps() -> ([char; 256], HashMap<char, u8>) {
    // Mirrors GPT-2's bytes_to_unicode table so arbitrary byte sequences can
    // flow through BPE merges without losing reversibility.
    let mut bs: Vec<u32> = (b'!'..=b'~').map(|b| b as u32).collect();
    bs.extend((0xA1u8..=0xAC).map(|b| b as u32));
    bs.extend((0xAEu8..=0xFF).map(|b| b as u32));

    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0u32..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut enc = ['\0'; 256];
    let mut dec = HashMap::with_capacity(256);
    for (b, c) in bs.into_iter().zip(cs.into_iter()) {
        if let Some(ch) = char::from_u32(c) {
            enc[b as usize] = ch;
            dec.insert(ch, b as u8);
        }
    }
    (enc, dec)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tekken splits every digit into its own pre-token. Grouping them, as the
    /// GPT-2 splitter does, silently changes the token count of every number.
    #[test]
    fn tekken_splits_digits_individually() {
        assert_eq!(pretokenize_tekken("2026"), vec!["2", "0", "2", "6"]);
        assert_eq!(pretokenize_tekken("v2"), vec!["v", "2"]);
        // The GPT-2 splitter keeps the run together — the contrast is the point.
        assert_eq!(pretokenize_gpt2("2026"), vec!["2026"]);
    }

    /// At most one leading character attaches to a word; the rest of a
    /// whitespace run becomes its own pre-token.
    #[test]
    fn tekken_attaches_at_most_one_leading_space() {
        assert_eq!(pretokenize_tekken("hello world"), vec!["hello", " world"]);
        assert_eq!(pretokenize_tekken("a  b"), vec!["a", " ", " b"]);
    }

    /// A whitespace run ending in newlines is isolated as one pre-token.
    #[test]
    fn tekken_isolates_newline_runs() {
        assert_eq!(pretokenize_tekken("a\n\nb"), vec!["a", "\n\n", "b"]);
        assert_eq!(pretokenize_tekken("a \n b"), vec!["a", " \n", " b"]);
    }

    /// Case boundaries split ASCII words, matching llama.cpp's collapsed
    /// character classes — including that all-caps runs stay together.
    #[test]
    fn tekken_splits_on_ascii_case_boundaries() {
        // A boundary appears only where a lowercase run is followed by an
        // uppercase letter.
        assert_eq!(pretokenize_tekken("HelloWorld"), vec!["Hello", "World"]);
        // An uppercase run is absorbed by the following lowercase run, so
        // neither of these splits — the greedy `UPPER* LOWER+` takes it all.
        assert_eq!(pretokenize_tekken("ABCd"), vec!["ABCd"]);
        assert_eq!(pretokenize_tekken("HTTPServer"), vec!["HTTPServer"]);
        // All-caps with no trailing lowercase falls to the `UPPER+ LOWER*`
        // branch and stays whole.
        assert_eq!(pretokenize_tekken("ABC"), vec!["ABC"]);
        // A non-ASCII letter satisfies both classes, so it never *causes* a
        // boundary — the uppercase run runs straight through the Cyrillic and
        // into the following ASCII capital. This mirrors llama.cpp's collapsed
        // classes; the stricter upstream regex would split these in two.
        assert_eq!(
            pretokenize_tekken("\u{041C}\u{043E}\u{0441}\u{043A}\u{0432}\u{0430}Moscow"),
            vec!["\u{041C}\u{043E}\u{0441}\u{043A}\u{0432}\u{0430}Moscow"]
        );
        // An ASCII capital still ends the preceding lowercase run, even when
        // that run contains non-ASCII letters.
        assert_eq!(pretokenize_tekken("MünchenBonn"), vec!["München", "Bonn"]);
    }

    /// Punctuation groups with one optional leading space and trailing slashes.
    #[test]
    fn tekken_groups_punctuation_runs() {
        assert_eq!(pretokenize_tekken("a, b"), vec!["a", ",", " b"]);
        assert_eq!(pretokenize_tekken("x = 1"), vec!["x", " =", " ", "1"]);
    }

    /// Every pre-tokenizer must reproduce its input exactly when concatenated;
    /// a dropped or duplicated character would corrupt the encoding silently.
    #[test]
    fn tekken_pretokenization_is_lossless() {
        for sample in [
            "Hello, world!",
            "  leading and trailing  ",
            "Zahl 2026 und \u{00E9}\u{00E9}",
            "\n\n\ttabs\r\nCRLF",
            "emoji \u{1F680} end",
            "mixedCASEandDigits123",
            "",
        ] {
            let joined: String = pretokenize_tekken(sample).concat();
            assert_eq!(joined, sample, "lossy split for {:?}", sample);
        }
    }

    /// A character split across two tokens must survive streaming decode.
    /// Decoding each token on its own yields two U+FFFD instead.
    #[test]
    fn utf8_stitcher_rejoins_split_code_points() {
        let rocket = "\u{1F680}".as_bytes().to_vec(); // 4 bytes
        let mut stitcher = Utf8Stitcher::new();
        assert_eq!(stitcher.push(&rocket[..1]), "");
        assert_eq!(stitcher.push(&rocket[1..3]), "");
        assert!(stitcher.has_pending());
        assert_eq!(stitcher.push(&rocket[3..]), "\u{1F680}");
        assert!(!stitcher.has_pending());
        assert_eq!(stitcher.flush(), "");
    }

    /// Text surrounding a split character must not be delayed or duplicated.
    #[test]
    fn utf8_stitcher_emits_complete_prefix_immediately() {
        let mixed = "ab\u{00E9}cd".as_bytes().to_vec();
        let mut stitcher = Utf8Stitcher::new();
        // Cut in the middle of the two-byte 'é'.
        let split = 3;
        let first = stitcher.push(&mixed[..split]);
        let second = stitcher.push(&mixed[split..]);
        assert_eq!(first, "ab");
        assert_eq!(second, "\u{00E9}cd");
        assert_eq!(format!("{}{}", first, second), "ab\u{00E9}cd");
    }

    /// A truncated tail at end of generation surfaces as U+FFFD rather than
    /// disappearing, and genuinely invalid bytes do not stall the stream.
    #[test]
    fn utf8_stitcher_flushes_and_recovers_from_invalid_bytes() {
        let mut stitcher = Utf8Stitcher::new();
        assert_eq!(stitcher.push(&[0xF0, 0x9F]), "");
        assert_eq!(stitcher.flush(), "\u{FFFD}");
        assert!(!stitcher.has_pending());

        // 0xFF can never begin a valid sequence, so it is reported at once and
        // the following ASCII still comes through.
        let mut stitcher = Utf8Stitcher::new();
        assert_eq!(stitcher.push(&[b'a', 0xFF, b'b']), "a\u{FFFD}b");
        assert!(!stitcher.has_pending());
    }

    /// The rescan-and-remove loop that [`merge_symbols`] replaced, kept as the
    /// reference its output is checked against. Strict `>` means the leftmost
    /// pair wins a tie, exactly as both original loops behaved.
    fn naive_merge<S, K, F>(symbols: &mut Vec<S>, mut pair: F)
    where
        K: Ord,
        F: FnMut(&S, &S) -> Option<(K, S)>,
    {
        while symbols.len() > 1 {
            let mut best: Option<(K, usize, S)> = None;
            for index in 0..symbols.len() - 1 {
                if let Some((key, merged)) = pair(&symbols[index], &symbols[index + 1]) {
                    let improves = best.as_ref().is_none_or(|(best_key, _, _)| key > *best_key);
                    if improves {
                        best = Some((key, index, merged));
                    }
                }
            }
            let Some((_, index, merged)) = best else {
                break;
            };
            symbols[index] = merged;
            symbols.remove(index + 1);
        }
    }

    /// Deterministic xorshift, so a failure is reproducible.
    fn next_random(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// The heap engine must agree with the scan on every input, including the
    /// cases that make staleness tricky: repeated symbols, chained merges, and
    /// pairs that tie on rank so only position separates them.
    #[test]
    fn merge_symbols_matches_naive_scan() {
        let mut state = 0x9E3779B97F4A7C15u64;
        for trial in 0..2000 {
            let length = 1 + (next_random(&mut state) % 40) as usize;
            let symbols: Vec<u32> = (0..length)
                .map(|_| (next_random(&mut state) % 8) as u32)
                .collect();

            // A deliberately dense table so long merge chains actually happen,
            // with ranks drawn from a small set to force ties between pairs.
            let mut table: HashMap<(u32, u32), (usize, u32)> = HashMap::new();
            for left in 0..12u32 {
                for right in 0..12u32 {
                    if next_random(&mut state) % 3 == 0 {
                        let rank = (next_random(&mut state) % 6) as usize;
                        let merged = (next_random(&mut state) % 12) as u32;
                        table.insert((left, right), (rank, merged));
                    }
                }
            }

            let rank_pair = |left: &u32, right: &u32| {
                table
                    .get(&(*left, *right))
                    .map(|&(rank, merged)| (Reverse(rank), merged))
            };
            let mut fast = symbols.clone();
            merge_symbols(&mut fast, rank_pair);
            let mut reference = symbols.clone();
            naive_merge(&mut reference, rank_pair);
            assert_eq!(fast, reference, "rank trial {trial} on {symbols:?}");

            // The same table read as scores exercises the float ordering and
            // the opposite sense of "better".
            let score_pair = |left: &u32, right: &u32| {
                table
                    .get(&(*left, *right))
                    .map(|&(rank, merged)| (ScoreKey(rank as f32), merged))
            };
            let mut fast = symbols.clone();
            merge_symbols(&mut fast, score_pair);
            let mut reference = symbols.clone();
            naive_merge(&mut reference, score_pair);
            assert_eq!(fast, reference, "score trial {trial} on {symbols:?}");
        }
    }

    fn str_array(items: &[&str]) -> MetaValue {
        MetaValue::Array(
            items
                .iter()
                .map(|item| MetaValue::Str((*item).to_string()))
                .collect(),
        )
    }

    fn gemma4_test_tokenizer() -> Tokenizer {
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            MetaValue::Str("gemma4".to_string()),
        );
        metadata.insert("tokenizer.ggml.bos_token_id".to_string(), MetaValue::U32(0));
        metadata.insert("tokenizer.ggml.eos_token_id".to_string(), MetaValue::U32(1));
        metadata.insert(
            "tokenizer.ggml.add_bos_token".to_string(),
            MetaValue::Bool(true),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            str_array(&[
                "<bos>",
                "<eos>",
                "H",
                "i",
                "\u{2581}",
                "t",
                "h",
                "e",
                "\n",
                "\n\n",
                "Hi",
                "\u{2581}t",
                "\u{2581}th",
                "\u{2581}the",
                "<0x21>",
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.scores".to_string(),
            MetaValue::Array(vec![MetaValue::F32(0.0); 15]),
        );
        metadata.insert(
            "tokenizer.ggml.merges".to_string(),
            str_array(&["H i", "\u{2581} t", "\u{2581}t h", "\u{2581}th e"]),
        );
        Tokenizer::from_metadata(&metadata)
    }

    #[test]
    fn gemma4_bpe_uses_spm_spaces_and_merge_ranks() {
        let tok = gemma4_test_tokenizer();
        assert_eq!(tok.encode_without_bos("Hi the"), vec![10, 13]);
    }

    #[test]
    fn gemma4_bpe_keeps_newline_runs_and_byte_fallback() {
        let tok = gemma4_test_tokenizer();
        assert_eq!(tok.encode_without_bos("Hi\n\n!"), vec![10, 9, 14]);
    }

    #[test]
    fn gemma4_splitter_groups_newline_runs() {
        assert_eq!(
            split_gemma4_pieces("a\n\nb\n"),
            vec!["a", "\n\n", "b", "\n"]
        );
    }

    fn wordpiece_test_tokenizer() -> Tokenizer {
        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            MetaValue::Str("bert".to_string()),
        );
        // CLS=2, SEP=3, UNK=1 in this synthetic vocab.
        metadata.insert("tokenizer.ggml.bos_token_id".to_string(), MetaValue::U32(2));
        metadata.insert("tokenizer.ggml.eos_token_id".to_string(), MetaValue::U32(3));
        metadata.insert(
            "tokenizer.ggml.unknown_token_id".to_string(),
            MetaValue::U32(1),
        );
        metadata.insert(
            "tokenizer.ggml.seperator_token_id".to_string(),
            MetaValue::U32(3),
        );
        metadata.insert(
            "tokenizer.ggml.add_bos_token".to_string(),
            MetaValue::Bool(true),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            // Phantom-space vocab: word-start pieces prefixed with U+2581,
            // continuation pieces stored bare.
            str_array(&[
                "[PAD]",         // 0
                "[UNK]",         // 1
                "[CLS]",         // 2
                "[SEP]",         // 3
                "\u{2581}hello", // 4
                "\u{2581}wor",   // 5
                "ld",            // 6
                "\u{2581},",     // 7
                "\u{2581}!",     // 8
                "\u{2581}cafe",  // 9
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.scores".to_string(),
            MetaValue::Array(vec![MetaValue::F32(0.0); 10]),
        );
        Tokenizer::from_metadata(&metadata)
    }

    #[test]
    fn wordpiece_frames_with_cls_and_sep() {
        let tok = wordpiece_test_tokenizer();
        // "hello" → [CLS] ▁hello [SEP]
        assert_eq!(tok.encode("hello"), vec![2, 4, 3]);
    }

    #[test]
    fn wordpiece_lowercases_and_isolates_punctuation() {
        let tok = wordpiece_test_tokenizer();
        // "Hello, world!" → CLS ▁hello ▁, ▁wor ld ▁! SEP
        assert_eq!(tok.encode("Hello, world!"), vec![2, 4, 7, 5, 6, 8, 3]);
    }

    #[test]
    fn wordpiece_greedy_longest_match() {
        let tok = wordpiece_test_tokenizer();
        // "world" splits into ▁wor + ld (greedy longest at each position).
        assert_eq!(tok.encode_without_bos("world"), vec![5, 6]);
    }

    #[test]
    fn wordpiece_unmatched_word_is_unk() {
        let tok = wordpiece_test_tokenizer();
        // "xyz" has no matching pieces → single [UNK].
        assert_eq!(tok.encode_without_bos("xyz"), vec![1]);
    }

    #[test]
    fn wordpiece_strips_accents() {
        let tok = wordpiece_test_tokenizer();
        // "Café" → deaccent+lowercase "cafe" → ▁cafe.
        assert_eq!(tok.encode_without_bos("Café"), vec![9]);
    }

    #[test]
    fn wordpiece_split_normalizes() {
        assert_eq!(
            wordpiece_split("Hello, WORLD!"),
            vec!["hello", ",", "world", "!"]
        );
    }

    /// Builds a SentencePiece tokenizer whose vocabulary is harvested from a
    /// corpus, so merge chains are as deep as a real model's rather than
    /// bottoming out after one step.
    fn spm_bench_tokenizer(corpus: &str) -> Tokenizer {
        let mut tokens: Vec<String> = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
        let mut seen: std::collections::HashSet<String> = tokens.iter().cloned().collect();
        for byte in 0u16..=255 {
            let piece = format!("<0x{:02X}>", byte);
            if seen.insert(piece.clone()) {
                tokens.push(piece);
            }
        }
        let chars: Vec<char> = corpus.replace(' ', "\u{2581}").chars().collect();
        for width in 1..=8usize {
            for window in chars.windows(width) {
                let piece: String = window.iter().collect();
                if seen.insert(piece.clone()) {
                    tokens.push(piece);
                }
            }
        }
        // SentencePiece prefers longer pieces, so score by length.
        let scores: Vec<MetaValue> = tokens
            .iter()
            .map(|t| MetaValue::F32(t.chars().count() as f32))
            .collect();

        let mut metadata = HashMap::new();
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            MetaValue::Str("llama".to_string()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            MetaValue::Array(tokens.into_iter().map(MetaValue::Str).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.scores".to_string(),
            MetaValue::Array(scores),
        );
        metadata.insert("tokenizer.ggml.bos_token_id".to_string(), MetaValue::U32(1));
        metadata.insert("tokenizer.ggml.eos_token_id".to_string(), MetaValue::U32(2));
        Tokenizer::from_metadata(&metadata)
    }

    const BENCH_CORPUS: &str = "Der Zug nach Muenchen faehrt um sieben Uhr ab und erreicht \
        den Hauptbahnhof gegen Mittag. Die Fahrkarte kostet vierzig Euro, eine Reservierung \
        ist optional. Reisende mit Gepaeck sollten den vorderen Wagen waehlen. The quick brown \
        fox jumps over the lazy dog while the engineer profiles a tokenizer merge loop and \
        wonders why long prompts feel slow to process on a laptop. ";

    /// End-to-end guard for the SentencePiece path: the heap engine has to be
    /// wired to the same scores, the same concatenation and the same `-inf`
    /// rejection as the scan, not merely be a correct engine in isolation.
    #[test]
    fn spm_encode_matches_naive_scan() {
        let tok = spm_bench_tokenizer(BENCH_CORPUS);
        for text in [
            "Der Zug nach Muenchen faehrt um sieben Uhr ab.",
            "The quick brown fox jumps over the lazy dog",
            "Reisende mit Gepaeck sollten den vorderen Wagen waehlen, sonst wird es eng.",
            "unbekannte Woerter wie Xylophonbauer kosten Bytes",
            "a",
            "  doppelte   Leerzeichen  ",
        ] {
            let mut reference = Vec::new();
            tok.encode_piece("\u{2581}", &mut reference);
            for ch in text.chars() {
                if ch == ' ' {
                    tok.encode_piece("\u{2581}", &mut reference);
                } else {
                    let mut buf = [0u8; 4];
                    tok.encode_piece(ch.encode_utf8(&mut buf), &mut reference);
                }
            }
            let mut merged = String::new();
            naive_merge(&mut reference, |&left: &u32, &right: &u32| {
                merged.clear();
                merged.push_str(tok.decode_raw(left));
                merged.push_str(tok.decode_raw(right));
                let id = *tok.token_to_id.get(merged.as_str())?;
                let score = tok.scores.get(id as usize).copied().unwrap_or(0.0);
                (score > f32::NEG_INFINITY).then_some((ScoreKey(score), id))
            });

            assert_eq!(
                tok.encode_without_bos(text),
                reference,
                "mismatch on {text:?}"
            );
        }
    }

    /// Same guard for byte-level BPE, against a real vocabulary and merge table
    /// rather than a synthetic one.
    #[test]
    #[ignore = "needs RUSTY_LLM_BENCH_MODEL"]
    #[cfg(not(target_family = "wasm"))]
    fn gguf_encode_matches_naive_scan() {
        let Ok(path) = std::env::var("RUSTY_LLM_BENCH_MODEL") else {
            println!("set RUSTY_LLM_BENCH_MODEL to a GGUF path to run this");
            return;
        };
        let mmap = crate::mmap::MmapFile::open(&path).expect("open model");
        let gguf = crate::gguf::GGUFFile::parse_quiet(mmap.as_slice()).expect("parse model");
        let tok = Tokenizer::from_metadata(&gguf.metadata);
        assert!(
            tok.mode == TokenizerMode::Gpt2Bpe,
            "expects a byte-level BPE model"
        );

        for text in [
            BENCH_CORPUS,
            "HelloWorld 2026 -- MixedCASE/slashes\n\nund Umlaute: Grueße, Muenchen",
            "emoji \u{1F680}\u{1F680} und CJK \u{4E2D}\u{6587} mitten im Satz",
        ] {
            let mut reference = Vec::new();
            let pieces = if tok.tekken {
                pretokenize_tekken(text)
            } else {
                pretokenize_gpt2(text)
            };
            for piece in pieces {
                if tok.tekken {
                    let encoded: String = piece
                        .as_bytes()
                        .iter()
                        .map(|&byte| tok.byte_encoder[byte as usize])
                        .collect();
                    if let Some(&id) = tok.token_to_id.get(&encoded) {
                        reference.push(id);
                        continue;
                    }
                }
                let mut symbols = Vec::new();
                for &byte in piece.as_bytes() {
                    let ch = tok.byte_encoder[byte as usize];
                    let mut buf = [0u8; 4];
                    let symbol = ch.encode_utf8(&mut buf);
                    match tok.token_to_id.get(symbol) {
                        Some(&id) => symbols.push(id),
                        None => tok.encode_piece(symbol, &mut symbols),
                    }
                }
                naive_merge(&mut symbols, |&left: &u32, &right: &u32| {
                    tok.bpe_merges
                        .get(&(left, right))
                        .map(|&(rank, merged_id)| (Reverse(rank), merged_id))
                });
                reference.extend(symbols);
            }

            assert_eq!(
                tok.encode_without_bos(text),
                reference,
                "mismatch on {text:?}"
            );
        }
    }

    /// Reports how encode time scales with prompt length.
    ///
    /// The number that matters is the ratio between successive rows: a
    /// quadratic merge loop roughly quadruples when the input doubles, a
    /// heap-driven one only slightly more than doubles.
    #[test]
    #[ignore = "performance measurement, not a correctness check"]
    fn bench_encode_scaling() {
        let corpus = BENCH_CORPUS.repeat(8);
        let tok = spm_bench_tokenizer(&corpus);

        println!("\nSentencePiece encode scaling");
        let mut previous: Option<(usize, f64)> = None;
        for repeats in [1usize, 2, 4, 8, 16] {
            let text = BENCH_CORPUS.repeat(repeats);
            let start = std::time::Instant::now();
            let ids = tok.encode_without_bos(&text);
            let millis = start.elapsed().as_secs_f64() * 1e3;
            let ratio = previous.map_or(String::from("—"), |(_, prev): (usize, f64)| {
                format!("{:.2}x", millis / prev)
            });
            println!(
                "  {:>6} chars -> {:>5} tokens in {:>9.2} ms  ({} vs previous row)",
                text.len(),
                ids.len(),
                millis,
                ratio
            );
            previous = Some((text.len(), millis));
        }
    }

    /// Same measurement for the byte-level BPE path, using the real vocabulary
    /// and merge table of a local GGUF when one is pointed at.
    #[test]
    #[ignore = "performance measurement, needs RUSTY_LLM_BENCH_MODEL"]
    #[cfg(not(target_family = "wasm"))]
    fn bench_gguf_encode() {
        let Ok(path) = std::env::var("RUSTY_LLM_BENCH_MODEL") else {
            println!("set RUSTY_LLM_BENCH_MODEL to a GGUF path to run this");
            return;
        };
        let mmap = crate::mmap::MmapFile::open(&path).expect("open model");
        let gguf = crate::gguf::GGUFFile::parse_quiet(mmap.as_slice()).expect("parse model");
        let tok = Tokenizer::from_metadata(&gguf.metadata);

        println!("\nGGUF encode scaling ({})", path);
        let mut previous: Option<f64> = None;
        for repeats in [1usize, 2, 4, 8, 16, 32] {
            let text = BENCH_CORPUS.repeat(repeats);
            let start = std::time::Instant::now();
            let ids = tok.encode_without_bos(&text);
            let millis = start.elapsed().as_secs_f64() * 1e3;
            let ratio =
                previous.map_or(String::from("—"), |prev| format!("{:.2}x", millis / prev));
            println!(
                "  {:>6} chars -> {:>5} tokens in {:>9.2} ms  ({} vs previous row)",
                text.len(),
                ids.len(),
                millis,
                ratio
            );
            previous = Some(millis);
        }
    }
}
