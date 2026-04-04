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
}

pub struct Tokenizer {
    vocab: Vec<String>,
    scores: Vec<f32>,
    token_to_id: HashMap<String, u32>,
    merge_ranks: HashMap<(String, String), usize>,
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    mode: TokenizerMode,
    add_bos_token: bool,
    pub bos_id: u32,
    pub eos_id: u32,
}

impl Tokenizer {
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

        let mut merge_ranks = HashMap::new();
        if let Some(merges) = metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_string_array())
        {
            for (rank, merge) in merges.iter().enumerate() {
                if let Some((left, right)) = merge.split_once(' ') {
                    merge_ranks.insert((left.to_string(), right.to_string()), rank);
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
        let mode = if model.eq_ignore_ascii_case("gpt2")
            || pre.to_ascii_lowercase().contains("qwen")
            || pre.to_ascii_lowercase().contains("gpt")
        {
            TokenizerMode::Gpt2Bpe
        } else {
            TokenizerMode::SentencePiece
        };

        let (byte_encoder, byte_decoder) = build_byte_maps();

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
            merge_ranks,
            byte_encoder,
            byte_decoder,
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

    pub fn encode_without_bos(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();

        if text.is_empty() {
            return tokens;
        }

        let encoded = match self.mode {
            TokenizerMode::SentencePiece => self.encode_sentencepiece(text),
            TokenizerMode::Gpt2Bpe => self.encode_gpt2_bpe(text),
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

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn special_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    fn encode_sentencepiece(&self, text: &str) -> Vec<u32> {
        // SentencePiece models encode word starts with U+2581, so we inject a
        // leading space before splitting to preserve first-token behavior.
        let processed = format!(" {}", text);
        let processed = processed.replace(' ', "\u{2581}");
        let current_tokens = self.encode_from_pieces(processed.chars().map(|ch| ch.to_string()));

        // Iterative BPE merge: find best adjacent pair by score, merge it
        let mut current_tokens = current_tokens;
        loop {
            if current_tokens.len() < 2 {
                break;
            }

            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;
            let mut best_id = 0u32;

            for i in 0..current_tokens.len() - 1 {
                let merged = format!(
                    "{}{}",
                    self.decode_raw(current_tokens[i]),
                    self.decode_raw(current_tokens[i + 1])
                );
                if let Some(&id) = self.token_to_id.get(&merged) {
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

    fn encode_gpt2_bpe(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for piece in pretokenize_gpt2(text) {
            // GPT-2 style BPE operates on a reversible byte-level alphabet
            // before merge ranks are applied.
            let mut encoded = String::with_capacity(piece.len());
            for &byte in piece.as_bytes() {
                if let Some(&ch) = self.byte_encoder.get(&byte) {
                    encoded.push(ch);
                }
            }

            let mut symbols: Vec<String> = encoded.chars().map(|ch| ch.to_string()).collect();
            while symbols.len() > 1 {
                let mut best_rank = usize::MAX;
                let mut best_idx = None;
                for i in 0..symbols.len() - 1 {
                    let pair = (symbols[i].clone(), symbols[i + 1].clone());
                    if let Some(&rank) = self.merge_ranks.get(&pair) {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = Some(i);
                        }
                    }
                }

                let Some(i) = best_idx else { break };
                let merged = format!("{}{}", symbols[i], symbols[i + 1]);
                symbols[i] = merged;
                symbols.remove(i + 1);
            }

            for symbol in symbols {
                if let Some(&id) = self.token_to_id.get(&symbol) {
                    out.push(id);
                } else {
                    out.extend(self.encode_from_pieces([symbol]));
                }
            }
        }
        out
    }

    fn encode_from_pieces<I>(&self, pieces: I) -> Vec<u32>
    where
        I: IntoIterator<Item = String>,
    {
        let mut out = Vec::new();
        for piece in pieces {
            if let Some(&id) = self.token_to_id.get(&piece) {
                out.push(id);
            } else {
                for byte in piece.as_bytes() {
                    let byte_tok = format!("<0x{:02X}>", byte);
                    if let Some(&id) = self.token_to_id.get(&byte_tok) {
                        out.push(id);
                    }
                }
            }
        }
        out
    }

    fn decode_gpt2_bytes(&self, raw: &str) -> String {
        let mut bytes = Vec::with_capacity(raw.len());
        for ch in raw.chars() {
            if let Some(&b) = self.byte_decoder.get(&ch) {
                bytes.push(b);
            } else {
                bytes.extend_from_slice(ch.to_string().as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn pretokenize_gpt2(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut pieces = Vec::new();
    let mut i = 0usize;

    // This is a lightweight approximation of GPT-2's regex pre-tokenizer:
    // group leading whitespace with the following token and split runs of
    // letters, digits, and punctuation separately.
    while i < chars.len() {
        let start = i;
        let mut had_space = false;
        while i < chars.len() && chars[i].is_whitespace() {
            had_space = true;
            i += 1;
        }

        if i >= chars.len() {
            if had_space {
                pieces.push(chars[start..i].iter().collect());
            }
            break;
        }

        let mut j = i;
        let c = chars[i];
        if c.is_alphabetic() {
            while j < chars.len() && chars[j].is_alphabetic() {
                j += 1;
            }
        } else if c.is_numeric() {
            while j < chars.len() && chars[j].is_numeric() {
                j += 1;
            }
        } else {
            while j < chars.len()
                && !chars[j].is_whitespace()
                && !chars[j].is_alphabetic()
                && !chars[j].is_numeric()
            {
                j += 1;
            }
        }

        let piece_start = if had_space { start } else { i };
        pieces.push(chars[piece_start..j].iter().collect());
        i = j;
    }

    pieces
}

fn build_byte_maps() -> (HashMap<u8, char>, HashMap<char, u8>) {
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

    let mut enc = HashMap::with_capacity(256);
    let mut dec = HashMap::with_capacity(256);
    for (b, c) in bs.into_iter().zip(cs.into_iter()) {
        if let Some(ch) = char::from_u32(c) {
            enc.insert(b as u8, ch);
            dec.insert(ch, b as u8);
        }
    }
    (enc, dec)
}
