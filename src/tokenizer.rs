// tokenizer.rs — BPE tokenizer from GGUF metadata
//
// Supports SentencePiece-style BPE with merge scores, byte fallback,
// and special token handling (BOS/EOS).

use crate::gguf::MetaValue;
use std::collections::HashMap;

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
    bpe_text_merges: HashMap<(String, String), (usize, u32)>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    byte_token_ids: [Option<u32>; 256],
    mode: TokenizerMode,
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
        let mut bpe_text_merges = HashMap::new();
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
                            .insert((left.to_string(), right.to_string()), (rank, merged_id));
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
        let mode = if model.eq_ignore_ascii_case("gemma4") {
            TokenizerMode::Gemma4Bpe
        } else if model.eq_ignore_ascii_case("bert") {
            TokenizerMode::WordPiece
        } else if model.eq_ignore_ascii_case("gpt2")
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

    /// User-facing decode: ▁ → space, handle byte tokens
    pub fn decode_token(&self, id: u32) -> String {
        let raw = self.decode_raw(id);

        if self.mode == TokenizerMode::Gpt2Bpe {
            return self.decode_gpt2_bytes(raw);
        }

        // Handle byte tokens <0xHH>
        if raw.starts_with("<0x") && raw.ends_with('>') && raw.len() == 6 {
            if let Ok(byte) = u8::from_str_radix(&raw[3..5], 16) {
                return String::from(byte as char);
            }
        }

        raw.replace('\u{2581}', " ")
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

        // Iterative BPE merge: find best adjacent pair by score, merge it
        let mut merged = String::new();
        loop {
            if current_tokens.len() < 2 {
                break;
            }

            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;
            let mut best_id = 0u32;

            for i in 0..current_tokens.len() - 1 {
                merged.clear();
                merged.push_str(self.decode_raw(current_tokens[i]));
                merged.push_str(self.decode_raw(current_tokens[i + 1]));
                if let Some(&id) = self.token_to_id.get(merged.as_str()) {
                    let score = if (id as usize) < self.scores.len() {
                        self.scores[id as usize]
                    } else {
                        0.0
                    };
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                        best_id = id;
                    }
                }
            }

            if best_idx == usize::MAX {
                break;
            }

            current_tokens[best_idx] = best_id;
            current_tokens.remove(best_idx + 1);
        }
        current_tokens
    }

    /// Encodes text with byte-level GPT-2 BPE.
    fn encode_gpt2_bpe(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for piece in pretokenize_gpt2(text) {
            // GPT-2 style BPE operates on a reversible byte-level alphabet
            // before merge ranks are applied.
            let mut symbols = Vec::with_capacity(piece.len());
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

            while symbols.len() > 1 {
                let mut best_rank = usize::MAX;
                let mut best_idx = None;
                let mut best_id = 0u32;
                for i in 0..symbols.len() - 1 {
                    if let Some(&(rank, merged_id)) =
                        self.bpe_merges.get(&(symbols[i], symbols[i + 1]))
                    {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = Some(i);
                            best_id = merged_id;
                        }
                    }
                }

                let Some(i) = best_idx else { break };
                symbols[i] = best_id;
                symbols.remove(i + 1);
            }

            out.extend(symbols);
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

            let mut symbols: Vec<String> = piece.chars().map(|ch| ch.to_string()).collect();
            while symbols.len() > 1 {
                let mut best_rank = usize::MAX;
                let mut best_idx = None;
                let mut best_id = 0u32;

                for i in 0..symbols.len() - 1 {
                    if let Some(&(rank, merged_id)) = self
                        .bpe_text_merges
                        .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                    {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = Some(i);
                            best_id = merged_id;
                        }
                    }
                }

                let Some(i) = best_idx else { break };
                let merged = self.decode_raw(best_id).to_string();
                symbols[i] = merged;
                symbols.remove(i + 1);
            }

            for symbol in symbols {
                self.encode_piece(&symbol, &mut out);
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
        let chars: Vec<char> = prefixed.chars().collect();
        let n = chars.len();
        let mut pieces = Vec::new();
        let mut start = 0usize;
        while start < n {
            let mut matched: Option<u32> = None;
            let mut matched_end = start;
            // Longest-match: try the widest window first.
            let max_end = (start + self.max_wp_token_chars).min(n);
            let mut end = max_end;
            while end > start {
                let candidate: String = chars[start..end].iter().collect();
                if let Some(&id) = self.token_to_id.get(&candidate) {
                    matched = Some(id);
                    matched_end = end;
                    break;
                }
                end -= 1;
            }
            match matched {
                Some(id) => {
                    pieces.push(id);
                    start = matched_end;
                }
                None => {
                    // Unmatched position ⇒ the whole word is unknown.
                    out.push(self.unk_id);
                    return;
                }
            }
        }
        out.extend(pieces);
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

    /// Decodes GPT-2 byte-level token text back to UTF-8 where possible.
    fn decode_gpt2_bytes(&self, raw: &str) -> String {
        let mut bytes = Vec::with_capacity(raw.len());
        for ch in raw.chars() {
            if let Some(&b) = self.byte_decoder.get(&ch) {
                bytes.push(b);
            } else {
                let mut buf = [0u8; 4];
                bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
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
}
