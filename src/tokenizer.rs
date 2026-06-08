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
    pub bos_id: u32,
    pub eos_id: u32,
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
            bos_id,
            eos_id,
        }
    }

    /// BPE encode: start with character/byte tokens, then greedily merge
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        if self.add_bos_token {
            tokens.push(self.bos_id);
        }
        tokens.extend_from_slice(&self.encode_without_bos(text));
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
}
