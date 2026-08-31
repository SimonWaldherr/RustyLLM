// tokenizer.rs — BPE tokenizer from GGUF metadata
//
// Supports SentencePiece-style BPE with merge scores, byte fallback,
// and special token handling (BOS/EOS).

use crate::gguf::MetaValue;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::ops::Range;

/// Multiplicative hasher (the FxHash construction, as used by `rustc-hash`)
/// for the tokenizer's lookup tables. `token_to_id` and friends are probed
/// once per byte/char/window during encoding, and `HashMap`'s default SipHash
/// pays a DoS-resistance cost that is pure waste here: every key comes from
/// the fixed GGUF vocabulary at load time, never from untrusted input.
#[derive(Default)]
struct FxHasher(u64);

const FX_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn mix(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(FX_SEED);
    }
}

impl Hasher for FxHasher {
    fn write(&mut self, mut bytes: &[u8]) {
        while let Some(chunk) = bytes.get(..8) {
            self.mix(u64::from_ne_bytes(chunk.try_into().unwrap()));
            bytes = &bytes[8..];
        }
        if let Some(chunk) = bytes.get(..4) {
            self.mix(u32::from_ne_bytes(chunk.try_into().unwrap()) as u64);
            bytes = &bytes[4..];
        }
        if let Some(chunk) = bytes.get(..2) {
            self.mix(u16::from_ne_bytes(chunk.try_into().unwrap()) as u64);
            bytes = &bytes[2..];
        }
        if let Some(&byte) = bytes.first() {
            self.mix(byte as u64);
        }
    }

    fn write_u8(&mut self, i: u8) {
        self.mix(i as u64);
    }
    fn write_u16(&mut self, i: u16) {
        self.mix(i as u64);
    }
    fn write_u32(&mut self, i: u32) {
        self.mix(i as u64);
    }
    fn write_u64(&mut self, i: u64) {
        self.mix(i);
    }
    fn write_usize(&mut self, i: usize) {
        self.mix(i as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

type FxBuildHasher = BuildHasherDefault<FxHasher>;

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
    token_to_id: HashMap<String, u32, FxBuildHasher>,
    bpe_merges: HashMap<(u32, u32), (usize, u32), FxBuildHasher>,
    /// Text-keyed merges for Gemma 4, nested left-then-right so a pair can be
    /// looked up from two `&str` without building an owned key for each probe.
    bpe_text_merges: HashMap<String, HashMap<String, (usize, u32), FxBuildHasher>, FxBuildHasher>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8, FxBuildHasher>,
    byte_token_ids: [Option<u32>; 256],
    /// Vocab id of each byte's own remapped single character, i.e. the common
    /// hit case of `encode_piece` precomputed as a table. `encode_gpt2_bpe`
    /// probes this once per input byte, so an array index instead of a
    /// hashmap lookup there is the difference between O(1) and paying a hash
    /// on every byte of every prompt.
    single_byte_ids: [Option<u32>; 256],
    /// Vocab id of each single ASCII character, for the SentencePiece symbol
    /// list. That list starts as one symbol per character of the *whole*
    /// prompt, so this turns the build from a hash lookup per character into
    /// an array index. `None` falls back to `encode_piece`'s byte fallback.
    spm_ascii_ids: [Option<u32>; 128],
    /// Vocab id of the SentencePiece word-start marker `\u{2581}`, which the
    /// symbol list emits once per space plus once at the front.
    spm_space_id: Option<u32>,
    mode: TokenizerMode,
    /// Mistral's Tekken pre-tokenizer: same byte-level vocabulary as GPT-2, but
    /// a different split and a whole-word vocabulary shortcut before merges.
    tekken: bool,
    /// Qwen3.5/3.6/3.8 use a distinct byte-level BPE splitter. In particular,
    /// numbers are individual pieces and combining marks stay with words.
    qwen35: bool,
    add_bos_token: bool,
    /// Longest vocab entry in chars; bounds the WordPiece greedy match window.
    max_wp_token_chars: usize,
    pub bos_id: u32,
    pub eos_id: u32,
    /// WordPiece unknown-token id (`tokenizer.ggml.unknown_token_id`).
    pub unk_id: u32,
    /// WordPiece separator id (`tokenizer.ggml.seperator_token_id`, preserving
    /// the legacy misspelled metadata key), appended after the encoded pieces.
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

        let mut token_to_id =
            HashMap::with_capacity_and_hasher(vocab.len(), FxBuildHasher::default());
        for (i, tok) in vocab.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as u32);
        }

        let mut bpe_merges: HashMap<(u32, u32), (usize, u32), FxBuildHasher> = HashMap::default();
        let mut bpe_text_merges: HashMap<
            String,
            HashMap<String, (usize, u32), FxBuildHasher>,
            FxBuildHasher,
        > = HashMap::default();
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
        let qwen35 = pre_lower.contains("qwen35");
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
        let mut single_byte_ids = [None; 256];
        for byte in 0u16..=255 {
            let byte_tok = format!("<0x{:02X}>", byte);
            byte_token_ids[byte as usize] = token_to_id.get(&byte_tok).copied();

            let ch = byte_encoder[byte as usize];
            let mut buf = [0u8; 4];
            let symbol = ch.encode_utf8(&mut buf);
            single_byte_ids[byte as usize] = token_to_id.get(symbol).copied();
        }

        let mut spm_ascii_ids = [None; 128];
        for (code, slot) in spm_ascii_ids.iter_mut().enumerate() {
            let mut buf = [0u8; 4];
            let symbol = (code as u8 as char).encode_utf8(&mut buf);
            *slot = token_to_id.get(symbol).copied();
        }
        let spm_space_id = token_to_id.get("\u{2581}").copied();

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
            single_byte_ids,
            spm_ascii_ids,
            spm_space_id,
            mode,
            tekken,
            qwen35,
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
    /// `[CLS] … [SEP]`, following the model's BERT framing metadata.
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
        // `push_spm_char` resolves the overwhelmingly common cases — an ASCII
        // character, or the word-start marker — from a precomputed table, and
        // only falls back to the hashing path for the rest.
        let push_spm_char = |ch: char, out: &mut Vec<u32>| {
            if ch == ' ' {
                match self.spm_space_id {
                    Some(id) => out.push(id),
                    None => self.encode_piece("\u{2581}", out),
                }
                return;
            }
            if ch.is_ascii()
                && let Some(id) = self.spm_ascii_ids[ch as usize]
            {
                out.push(id);
                return;
            }
            let mut buf = [0u8; 4];
            self.encode_piece(ch.encode_utf8(&mut buf), out);
        };

        push_spm_char(' ', &mut current_tokens);
        for ch in text.chars() {
            push_spm_char(ch, &mut current_tokens);
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
        let pieces = if self.qwen35 {
            pretokenize_qwen35(text)
        } else if self.tekken {
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
                if let Some(id) = self.single_byte_ids[byte as usize] {
                    symbols.push(id);
                } else {
                    let ch = self.byte_encoder[byte as usize];
                    let mut buf = [0u8; 4];
                    let symbol = ch.encode_utf8(&mut buf);
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
    /// Normalizes by lowercasing, stripping accents, and dropping control
    /// characters; then splits whitespace, isolates punctuation, ASCII symbols,
    /// and CJK as single-character words, and greedily matches the longest
    /// vocab piece at each position. The GGUF vocab is phantom-space form, so
    /// each word is prefixed with `\u{2581}` and word-start pieces are matched
    /// with that prefix while continuation pieces are matched bare. Any word
    /// with a position that fails to match becomes a single `[UNK]`.
    fn encode_wordpiece(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        // Every buffer here is reused for the whole text. The straightforward
        // version costs three allocations per word — the word itself, its
        // phantom-space copy, and its offset table — which for an embedding
        // input of a few hundred words is most of the work this path does.
        let mut normalized = String::new();
        let mut words = Vec::new();
        wordpiece_split_into(text, &mut normalized, &mut words);

        let mut prefixed = String::new();
        let mut offsets = Vec::new();
        for word in &words {
            self.encode_wordpiece_word(
                &normalized[word.clone()],
                &mut out,
                &mut prefixed,
                &mut offsets,
            );
        }
        out
    }

    /// Greedily segments one normalized word into WordPiece token ids.
    ///
    /// `prefixed` and `offsets` are caller-owned scratch; their contents on
    /// entry are irrelevant and they exist only so the allocation is not
    /// repeated per word.
    fn encode_wordpiece_word(
        &self,
        word: &str,
        out: &mut Vec<u32>,
        prefixed: &mut String,
        offsets: &mut Vec<usize>,
    ) {
        if word.is_empty() {
            return;
        }
        // Phantom-space form: a leading `\u{2581}` marks the word start, and
        // continuation pieces are stored without the `##` prefix.
        prefixed.clear();
        prefixed.push('\u{2581}');
        prefixed.push_str(word);
        // Char-start byte offsets let every candidate window be a borrowed
        // slice; collecting the chars instead would rebuild a `String` for each
        // of the (up to `max_wp_token_chars`) windows tried per position.
        offsets.clear();
        offsets.extend(prefixed.char_indices().map(|(at, _)| at));
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
/// Uses the compatibility TEKKEN regex where every non-ASCII letter belongs to
/// one class, so case boundaries only exist for `a-z`/`A-Z`. Preserving that
/// behavior keeps token counts stable across supported GGUF tokenizers and
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

/// Splits text with Qwen3.5/3.6/3.8's byte-level BPE pattern.
///
/// Qwen35 differs from ordinary GPT-2 BPE in ways that affect prompt IDs:
/// numerical code points are individual pieces, contractions are isolated,
/// and combining marks remain with their word. The implementation mirrors the
/// tokenizer.json pattern without pulling a regex engine into the hot path:
///
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
fn pretokenize_qwen35(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let offset = |index: usize| chars.get(index).map_or(text.len(), |(byte, _)| *byte);
    let mut pieces = Vec::with_capacity(chars.len() / 3 + 1);
    let mut index = 0usize;

    while index < chars.len() {
        if let Some(end) = qwen35_contraction_end(&chars, index) {
            pieces.push(&text[offset(index)..offset(end)]);
            index = end;
            continue;
        }
        if let Some(end) = qwen35_word_end(&chars, index) {
            pieces.push(&text[offset(index)..offset(end)]);
            index = end;
            continue;
        }
        if chars[index].1.is_numeric() {
            pieces.push(&text[offset(index)..offset(index + 1)]);
            index += 1;
            continue;
        }
        if let Some(end) = qwen35_punctuation_end(&chars, index) {
            pieces.push(&text[offset(index)..offset(end)]);
            index = end;
            continue;
        }
        if chars[index].1.is_whitespace() {
            let end = qwen35_whitespace_end(&chars, index);
            pieces.push(&text[offset(index)..offset(end)]);
            index = end;
            continue;
        }
        pieces.push(&text[offset(index)..offset(index + 1)]);
        index += 1;
    }

    pieces
}

#[inline]
fn is_qwen35_word_char(ch: char) -> bool {
    ch.is_alphabetic() || is_combining_mark(ch)
}

/// Unicode's Mark categories are not exposed by Rust's stable `char` API.
/// Keep the common combining blocks explicit so diacritics stay with their
/// base word, rather than becoming punctuation-only BPE pieces.
#[inline]
fn is_combining_mark(ch: char) -> bool {
    // No combining block starts below U+0300, so ASCII and Latin-1 — the bulk
    // of real input, and every space, digit and punctuation mark in it — can
    // skip the range chain below outright. `is_qwen35_word_char` reaches this
    // for every non-alphabetic character of every Qwen prompt, so the early
    // exit is the difference between one comparison and ~200.
    if (ch as u32) < 0x0300 {
        return false;
    }
    matches!(
        ch as u32,
        0x0300..=0x036F
            | 0x0483..=0x0489
            | 0x0591..=0x05BD
            | 0x05BF
            | 0x05C1..=0x05C2
            | 0x05C4..=0x05C5
            | 0x05C7
            | 0x0610..=0x061A
            | 0x064B..=0x065F
            | 0x0670
            | 0x06D6..=0x06ED
            | 0x0711
            | 0x0730..=0x074A
            | 0x07A6..=0x07B0
            | 0x07EB..=0x07F3
            | 0x07FD
            | 0x0816..=0x0819
            | 0x081B..=0x0823
            | 0x0825..=0x0827
            | 0x0829..=0x082D
            | 0x0859..=0x085B
            | 0x0897..=0x089F
            | 0x08CA..=0x08E1
            | 0x08E3..=0x0903
            | 0x093A..=0x093C
            | 0x093E..=0x094F
            | 0x0951..=0x0957
            | 0x0962..=0x0963
            | 0x0981..=0x0983
            | 0x09BC
            | 0x09BE..=0x09C4
            | 0x09C7..=0x09C8
            | 0x09CB..=0x09CD
            | 0x09D7
            | 0x09E2..=0x09E3
            | 0x09FE
            | 0x0A01..=0x0A03
            | 0x0A3C
            | 0x0A3E..=0x0A42
            | 0x0A47..=0x0A48
            | 0x0A4B..=0x0A4D
            | 0x0A51
            | 0x0A70..=0x0A71
            | 0x0A75
            | 0x0A81..=0x0A83
            | 0x0ABC
            | 0x0ABE..=0x0AC5
            | 0x0AC7..=0x0AC9
            | 0x0ACB..=0x0ACD
            | 0x0AE2..=0x0AE3
            | 0x0AFA..=0x0AFF
            | 0x0B01..=0x0B03
            | 0x0B3C
            | 0x0B3E..=0x0B44
            | 0x0B47..=0x0B48
            | 0x0B4B..=0x0B4D
            | 0x0B55..=0x0B57
            | 0x0B62..=0x0B63
            | 0x0B82
            | 0x0BBE..=0x0BC2
            | 0x0BC6..=0x0BC8
            | 0x0BCA..=0x0BCD
            | 0x0BD7
            | 0x0C00..=0x0C04
            | 0x0C3C
            | 0x0C3E..=0x0C44
            | 0x0C46..=0x0C48
            | 0x0C4A..=0x0C4D
            | 0x0C55..=0x0C56
            | 0x0C62..=0x0C63
            | 0x0C81..=0x0C83
            | 0x0CBC
            | 0x0CBE..=0x0CC4
            | 0x0CC6..=0x0CC8
            | 0x0CCA..=0x0CCD
            | 0x0CD5..=0x0CD6
            | 0x0CE2..=0x0CE3
            | 0x0CF3
            | 0x0D00..=0x0D03
            | 0x0D3B..=0x0D3C
            | 0x0D3E..=0x0D44
            | 0x0D46..=0x0D48
            | 0x0D4A..=0x0D4D
            | 0x0D57
            | 0x0D62..=0x0D63
            | 0x0D81..=0x0D83
            | 0x0DCA
            | 0x0DCF..=0x0DD4
            | 0x0DD6
            | 0x0DD8..=0x0DDF
            | 0x0DF2..=0x0DF3
            | 0x0E31
            | 0x0E34..=0x0E3A
            | 0x0E47..=0x0E4E
            | 0x0EB1
            | 0x0EB4..=0x0EBC
            | 0x0EC8..=0x0ECE
            | 0x0F18..=0x0F19
            | 0x0F35
            | 0x0F37
            | 0x0F39
            | 0x0F3E..=0x0F3F
            | 0x0F71..=0x0F84
            | 0x0F86..=0x0F87
            | 0x0F8D..=0x0F97
            | 0x0F99..=0x0FBC
            | 0x0FC6
            | 0x102B..=0x103E
            | 0x1056..=0x1059
            | 0x105E..=0x1060
            | 0x1062..=0x1064
            | 0x1067..=0x106D
            | 0x1071..=0x1074
            | 0x1082..=0x108D
            | 0x108F
            | 0x109A..=0x109D
            | 0x135D..=0x135F
            | 0x1712..=0x1715
            | 0x1732..=0x1734
            | 0x1752..=0x1753
            | 0x1772..=0x1773
            | 0x17B4..=0x17D3
            | 0x17DD
            | 0x180B..=0x180D
            | 0x180F
            | 0x1885..=0x1886
            | 0x18A9
            | 0x1920..=0x192B
            | 0x1930..=0x193B
            | 0x1A17..=0x1A1B
            | 0x1A55..=0x1A5E
            | 0x1A60..=0x1A7C
            | 0x1A7F
            | 0x1AB0..=0x1ACE
            | 0x1B00..=0x1B04
            | 0x1B34..=0x1B44
            | 0x1B6B..=0x1B73
            | 0x1B80..=0x1B82
            | 0x1BA1..=0x1BAD
            | 0x1BE6..=0x1BF3
            | 0x1C24..=0x1C37
            | 0x1CD0..=0x1CD2
            | 0x1CD4..=0x1CE8
            | 0x1CED
            | 0x1CF4
            | 0x1CF7..=0x1CF9
            | 0x1DC0..=0x1DFF
            | 0x20D0..=0x20F0
            | 0x2CEF..=0x2CF1
            | 0x2D7F
            | 0x2DE0..=0x2DFF
            | 0x302A..=0x302F
            | 0x3099..=0x309A
            | 0xA66F..=0xA672
            | 0xA674..=0xA67D
            | 0xA69E..=0xA69F
            | 0xA6F0..=0xA6F1
            | 0xA802
            | 0xA806
            | 0xA80B
            | 0xA823..=0xA827
            | 0xA82C
            | 0xA880..=0xA881
            | 0xA8B4..=0xA8C5
            | 0xA8E0..=0xA8F1
            | 0xA8FF
            | 0xA926..=0xA92D
            | 0xA947..=0xA953
            | 0xA980..=0xA983
            | 0xA9B3..=0xA9C0
            | 0xA9E5
            | 0xAA29..=0xAA36
            | 0xAA43
            | 0xAA4C..=0xAA4D
            | 0xAA7B..=0xAA7D
            | 0xAAB0
            | 0xAAB2..=0xAAB4
            | 0xAAB7..=0xAAB8
            | 0xAABE..=0xAABF
            | 0xAAC1
            | 0xAAEB..=0xAAEF
            | 0xAAF5..=0xAAF6
            | 0xABE3..=0xABEA
            | 0xABEC..=0xABED
            | 0xFB1E
            | 0xFE00..=0xFE0F
            | 0xFE20..=0xFE2F
            | 0x101FD
            | 0x102E0
            | 0x10376..=0x1037A
            | 0x10A01..=0x10A03
            | 0x10A05..=0x10A06
            | 0x10A0C..=0x10A0F
            | 0x10A38..=0x10A3A
            | 0x10A3F
            | 0x10AE5..=0x10AE6
            | 0x10D24..=0x10D27
            | 0x10D69..=0x10D6D
            | 0x10EAB..=0x10EAC
            | 0x10EFC..=0x10EFF
            | 0x10F46..=0x10F50
            | 0x10F82..=0x10F85
            | 0x11000..=0x11002
            | 0x11038..=0x11046
            | 0x11070
            | 0x11073..=0x11074
            | 0x1107F..=0x11082
            | 0x110B0..=0x110BA
            | 0x110C2
            | 0x11100..=0x11102
            | 0x11127..=0x11134
            | 0x11145..=0x11146
            | 0x11173
            | 0x11180..=0x11182
            | 0x111B3..=0x111C0
            | 0x111C9..=0x111CC
            | 0x111CE..=0x111CF
            | 0x1122C..=0x11237
            | 0x1123E
            | 0x11241
            | 0x112DF..=0x112EA
            | 0x11300..=0x11303
            | 0x1133B..=0x1133C
            | 0x1133E..=0x11344
            | 0x11347..=0x11348
            | 0x1134B..=0x1134D
            | 0x11357
            | 0x11362..=0x11363
            | 0x11366..=0x1136C
            | 0x11370..=0x11374
            | 0x113B8..=0x113C0
            | 0x113C2
            | 0x113C5
            | 0x113C7..=0x113CA
            | 0x113CC..=0x113D0
            | 0x113D2
            | 0x113E1..=0x113E2
            | 0x11435..=0x11446
            | 0x1145E
            | 0x114B0..=0x114C3
            | 0x115AF..=0x115B5
            | 0x115B8..=0x115C0
            | 0x115DC..=0x115DD
            | 0x11630..=0x11640
            | 0x116AB..=0x116B7
            | 0x1171D..=0x1172B
            | 0x1182C..=0x1183A
            | 0x11930..=0x11935
            | 0x11937..=0x11938
            | 0x1193B..=0x1193E
            | 0x11940
            | 0x11942..=0x11943
            | 0x119D1..=0x119D7
            | 0x119DA..=0x119E0
            | 0x119E4
            | 0x11A01..=0x11A0A
            | 0x11A33..=0x11A39
            | 0x11A3B..=0x11A3E
            | 0x11A47
            | 0x11A51..=0x11A5B
            | 0x11A8A..=0x11A99
            | 0x11C2F..=0x11C36
            | 0x11C38..=0x11C3F
            | 0x11C92..=0x11CA7
            | 0x11CA9..=0x11CB6
            | 0x11D31..=0x11D36
            | 0x11D3A
            | 0x11D3C..=0x11D3D
            | 0x11D3F..=0x11D45
            | 0x11D47
            | 0x11D8A..=0x11D8E
            | 0x11D90..=0x11D91
            | 0x11D93..=0x11D97
            | 0x11EF3..=0x11EF6
            | 0x11F00..=0x11F01
            | 0x11F03
            | 0x11F34..=0x11F3A
            | 0x11F3E..=0x11F42
            | 0x11F5A
            | 0x13440
            | 0x13447..=0x13455
            | 0x1611E..=0x1612F
            | 0x16AF0..=0x16AF4
            | 0x16B30..=0x16B36
            | 0x16F4F
            | 0x16F51..=0x16F87
            | 0x16F8F..=0x16F92
            | 0x16FE4
            | 0x16FF0..=0x16FF1
            | 0x1BC9D..=0x1BC9E
            | 0x1CF00..=0x1CF2D
            | 0x1CF30..=0x1CF46
            | 0x1D165..=0x1D169
            | 0x1D16D..=0x1D172
            | 0x1D17B..=0x1D182
            | 0x1D185..=0x1D18B
            | 0x1D1AA..=0x1D1AD
            | 0x1D242..=0x1D244
            | 0x1DA00..=0x1DA36
            | 0x1DA3B..=0x1DA6C
            | 0x1DA75
            | 0x1DA84
            | 0x1DA9B..=0x1DA9F
            | 0x1DAA1..=0x1DAAF
            | 0x1E000..=0x1E006
            | 0x1E008..=0x1E018
            | 0x1E01B..=0x1E021
            | 0x1E023..=0x1E024
            | 0x1E026..=0x1E02A
            | 0x1E08F
            | 0x1E130..=0x1E136
            | 0x1E2AE
            | 0x1E2EC..=0x1E2EF
            | 0x1E4EC..=0x1E4EF
            | 0x1E5EE..=0x1E5EF
            | 0x1E8D0..=0x1E94A
            | 0xE0100..=0xE01EF
    )
}

fn qwen35_contraction_end(chars: &[(usize, char)], index: usize) -> Option<usize> {
    if chars.get(index).is_none_or(|(_, ch)| *ch != '\'') {
        return None;
    }
    for suffix in ["re", "ve", "ll", "s", "t", "m", "d"] {
        let end = index + 1 + suffix.len();
        if end > chars.len() {
            continue;
        }
        if suffix
            .chars()
            .enumerate()
            .all(|(offset, expected)| chars[index + 1 + offset].1.eq_ignore_ascii_case(&expected))
        {
            return Some(end);
        }
    }
    None
}

fn qwen35_word_end(chars: &[(usize, char)], index: usize) -> Option<usize> {
    let word_end = |start: usize| {
        let mut end = start;
        while end < chars.len() && is_qwen35_word_char(chars[end].1) {
            end += 1;
        }
        end
    };
    let end = word_end(index);
    if end > index {
        return Some(end);
    }
    if let Some((_, ch)) = chars.get(index)
        && *ch != '\r'
        && *ch != '\n'
        && !ch.is_alphabetic()
        && !ch.is_numeric()
    {
        let end = word_end(index + 1);
        if end > index + 1 {
            return Some(end);
        }
    }
    None
}

fn qwen35_punctuation_end(chars: &[(usize, char)], index: usize) -> Option<usize> {
    let mut end = index;
    if chars.get(end).is_some_and(|(_, ch)| *ch == ' ') {
        end += 1;
    }
    let start = end;
    while end < chars.len()
        && !chars[end].1.is_whitespace()
        && !is_qwen35_word_char(chars[end].1)
        && !chars[end].1.is_numeric()
    {
        end += 1;
    }
    if end == start {
        return None;
    }
    while end < chars.len() && matches!(chars[end].1, '\r' | '\n') {
        end += 1;
    }
    Some(end)
}

fn qwen35_whitespace_end(chars: &[(usize, char)], index: usize) -> usize {
    let mut end = index;
    let mut last_newline = None;
    while end < chars.len() && chars[end].1.is_whitespace() {
        if matches!(chars[end].1, '\r' | '\n') {
            last_newline = Some(end);
        }
        end += 1;
    }
    last_newline.map_or(end, |newline| newline + 1)
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
/// Applies WPM preprocessing: lowercase, strip accents, drop control and
/// replacement characters, split on Unicode whitespace, and emit
/// each punctuation char, ASCII symbol, or CJK char as its own word.
///
/// All words are appended to `buffer` and reported as byte ranges into it, so
/// a text of n words costs one growing allocation rather than n separate
/// `String`s. Both arguments are cleared on entry.
fn wordpiece_split_into(text: &str, buffer: &mut String, words: &mut Vec<Range<usize>>) {
    buffer.clear();
    words.clear();
    // Start of the word currently being accumulated at the end of `buffer`.
    let mut start = 0usize;
    // A word is only emitted once it has content, which also covers a
    // character that normalizes away to nothing.
    let flush = |buffer: &String, words: &mut Vec<Range<usize>>, start: &mut usize| {
        if buffer.len() > *start {
            words.push(*start..buffer.len());
            *start = buffer.len();
        }
    };

    for ch in text.chars() {
        // Drop the null and replacement characters and C0/C1 control chars.
        if ch == '\u{0}' || ch == '\u{fffd}' || ch.is_control() {
            continue;
        }
        if ch.is_whitespace() {
            flush(buffer, words, &mut start);
            continue;
        }
        if is_wordpiece_standalone(ch) {
            flush(buffer, words, &mut start);
            push_accent_stripped_lower(ch, buffer);
            flush(buffer, words, &mut start);
            continue;
        }
        push_accent_stripped_lower(ch, buffer);
    }
    flush(buffer, words, &mut start);
}

/// Reports whether a character is tokenized as its own WordPiece word
/// (punctuation, ASCII symbols, or CJK ideographs).
fn is_wordpiece_standalone(ch: char) -> bool {
    // Every range below sits above U+7F, and the trailing clause is itself
    // gated on `cp > 0x7F`, so for ASCII the whole function reduces to the
    // punctuation test. Answering it here keeps ASCII input — the common case
    // for embedding inputs — off the range chain.
    if ch.is_ascii() {
        return ch.is_ascii_punctuation();
    }
    let cp = ch as u32;
    // ASCII symbols that are not is_ascii_punctuation (none extra) plus the
    // Unicode CJK ideograph ranges isolated as individual tokens.
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
/// common case for nomic-embed inputs), appending straight into `out`. Every
/// character of the input passes through here, so writing in place instead of
/// returning a `String` avoids a heap allocation per character. Combining
/// marks are dropped; other scripts pass through lowercased. Full Unicode NFD
/// parity is out of scope, so rare non-Latin accents may remain model-specific.
fn push_accent_stripped_lower(ch: char, out: &mut String) {
    // ASCII has no combining marks and no decomposition in `latin_deaccent`
    // (whose arms are all non-ASCII), so it reduces to an ASCII lowercase —
    // without walking the deaccent match or `to_lowercase`'s Unicode tables.
    if ch.is_ascii() {
        out.push(ch.to_ascii_lowercase());
        return;
    }
    // Drop combining diacritical marks outright.
    if ('\u{0300}'..='\u{036F}').contains(&ch) {
        return;
    }
    let base = latin_deaccent(ch).unwrap_or(ch);
    out.extend(base.to_lowercase());
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
fn build_byte_maps() -> ([char; 256], HashMap<char, u8, FxBuildHasher>) {
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
    let mut dec = HashMap::with_capacity_and_hasher(256, FxBuildHasher::default());
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

    /// Qwen3.5's pre-tokenizer is deliberately distinct from Qwen2/GPT-2:
    /// it isolates each numeric code point and keeps a combining mark attached
    /// to the word it modifies. Both rules change the subsequent BPE merges.
    #[test]
    fn qwen35_splits_digits_and_keeps_combining_marks_with_words() {
        assert_eq!(pretokenize_qwen35("v2026"), vec!["v", "2", "0", "2", "6"]);
        assert_eq!(
            pretokenize_qwen35("cafe\u{0301} 42"),
            vec!["cafe\u{0301}", " ", "4", "2"]
        );
        // GPT-2 groups the numeric run, which is exactly the boundary Qwen35
        // must not inherit.
        assert_eq!(pretokenize_gpt2("v2026"), vec!["v", "2026"]);
    }

    /// The native Qwen35 expression handles contractions before ordinary
    /// words, and permits one literal leading space on a word or punctuation
    /// pre-token.
    #[test]
    fn qwen35_handles_contractions_and_leading_space() {
        assert_eq!(
            pretokenize_qwen35("don't I'LL!"),
            vec!["don", "'t", " I", "'LL", "!"]
        );
        assert_eq!(
            pretokenize_qwen35("hello, world!"),
            vec!["hello", ",", " world", "!"]
        );
    }

    /// Every pre-tokenizer must preserve the original UTF-8 byte sequence;
    /// this catches progress errors around newline and non-ASCII boundaries.
    #[test]
    fn qwen35_pretokenization_is_lossless() {
        for input in [
            "a\\n\\nb",
            "Hallo, Welt! 123",
            "e\u{0301} — नमस्ते ٤٢",
            "  trailing whitespace  ",
        ] {
            assert_eq!(pretokenize_qwen35(input).concat(), input);
        }
    }

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

    /// Case boundaries split ASCII words using collapsed character classes,
    /// including that all-caps runs stay together.
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
        // into the following ASCII capital. The compatibility classes keep the
        // run intact; a stricter Unicode case regex would split these in two.
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

    /// The normalization helpers short-circuit ASCII before consulting their
    /// Unicode range chains. A misplaced cutoff would not fail loudly — it
    /// would just retokenize ordinary text differently — so the boundaries are
    /// pinned here.
    #[test]
    fn ascii_fast_paths_match_general_logic() {
        // The cutoff must sit exactly on the first combining block.
        assert!(!is_combining_mark('\u{02FF}'));
        assert!(is_combining_mark('\u{0300}'));
        assert!(is_combining_mark('\u{036F}'));

        // Non-ASCII standalone characters must still be recognised past the
        // ASCII early return.
        assert!(is_wordpiece_standalone('\u{4E00}')); // CJK ideograph
        assert!(is_wordpiece_standalone('\u{2014}')); // em dash
        assert!(!is_wordpiece_standalone('\u{00E9}')); // é is a word character

        for code in 0u8..=0x7F {
            let ch = code as char;
            assert!(!is_combining_mark(ch), "U+{code:04X}");
            assert_eq!(
                is_wordpiece_standalone(ch),
                ch.is_ascii_punctuation(),
                "U+{code:04X}"
            );

            // The ASCII branch must produce what the deaccent + Unicode
            // lowercase path it skips would have produced.
            let mut fast = String::new();
            push_accent_stripped_lower(ch, &mut fast);
            let general: String = latin_deaccent(ch).unwrap_or(ch).to_lowercase().collect();
            assert_eq!(fast, general, "U+{code:04X}");
        }
    }

    #[test]
    fn wordpiece_split_normalizes() {
        let mut buffer = String::new();
        let mut words = Vec::new();
        wordpiece_split_into("Hello, WORLD!", &mut buffer, &mut words);
        let split: Vec<&str> = words.iter().map(|word| &buffer[word.clone()]).collect();
        assert_eq!(split, vec!["hello", ",", "world", "!"]);
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
