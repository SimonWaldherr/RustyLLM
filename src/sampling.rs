// sampling.rs — Token sampling strategies

/// Simple xorshift64 PRNG (fast, no dependencies)
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_1337
            } else {
                seed
            },
        }
    }

    /// Returns uniform f32 in [0, 1)
    pub fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[derive(Clone, Debug)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
        }
    }
}

pub fn sample(
    logits: &mut [f32],
    config: &SamplerConfig,
    rng: &mut Rng,
    recent_tokens: &[u32],
) -> u32 {
    let mut candidates = Vec::new();
    sample_with_scratch(logits, config, rng, recent_tokens, &mut candidates)
}

pub fn sample_with_scratch(
    logits: &mut [f32],
    config: &SamplerConfig,
    rng: &mut Rng,
    recent_tokens: &[u32],
    candidates: &mut Vec<(usize, f32)>,
) -> u32 {
    let n = logits.len();
    if n == 0 {
        return 0;
    }

    for v in logits.iter_mut() {
        if !v.is_finite() {
            *v = f32::NEG_INFINITY;
        }
    }

    // Repetition penalty
    if config.repeat_penalty != 1.0 {
        for &tok in recent_tokens {
            if (tok as usize) < n {
                if logits[tok as usize] > 0.0 {
                    logits[tok as usize] /= config.repeat_penalty;
                } else {
                    logits[tok as usize] *= config.repeat_penalty;
                }
            }
        }
    }

    // Greedy
    if config.temperature < 1e-6 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap();
    }

    // Temperature
    let inv_temp = 1.0 / config.temperature;
    for v in logits.iter_mut() {
        *v *= inv_temp;
    }

    if config.top_k > 0 && config.top_k < n {
        return sample_top_k(logits, config.top_k, config.top_p, rng, candidates);
    }

    // Softmax
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    let inv_sum = 1.0 / sum;
    for v in logits.iter_mut() {
        *v *= inv_sum;
    }

    // Top-P (nucleus sampling)
    if config.top_p < 1.0 {
        let mut sorted: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        sorted.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let mut cumsum = 0.0f32;
        let mut cutoff_idx = sorted.len();
        for (i, &(_, p)) in sorted.iter().enumerate() {
            cumsum += p;
            if cumsum > config.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out tokens below cutoff
        let mut keep = vec![false; n];
        for &(idx, _) in &sorted[..cutoff_idx] {
            keep[idx] = true;
        }
        let mut new_sum = 0.0f32;
        for i in 0..n {
            if !keep[i] {
                logits[i] = 0.0;
            } else {
                new_sum += logits[i];
            }
        }
        // Renormalize
        if new_sum > 0.0 {
            let inv = 1.0 / new_sum;
            for v in logits.iter_mut() {
                *v *= inv;
            }
        }
    }

    // Sample from distribution
    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in logits.iter().enumerate() {
        cumsum += p;
        if cumsum > r {
            return i as u32;
        }
    }

    // Fallback
    (n - 1) as u32
}

fn sample_top_k(
    logits: &[f32],
    top_k: usize,
    top_p: f32,
    rng: &mut Rng,
    candidates: &mut Vec<(usize, f32)>,
) -> u32 {
    candidates.clear();
    if candidates.capacity() < top_k {
        candidates.reserve(top_k - candidates.capacity());
    }

    for (idx, &logit) in logits.iter().enumerate() {
        if candidates.len() < top_k {
            candidates.push((idx, logit));
            bubble_up_last(candidates);
        } else if logit.total_cmp(&candidates[candidates.len() - 1].1).is_gt() {
            let last = candidates.len() - 1;
            candidates[last] = (idx, logit);
            bubble_up_last(candidates);
        }
    }

    if candidates.is_empty() || !candidates[0].1.is_finite() {
        return argmax_token(logits);
    }

    let max = candidates[0].1;
    let mut sum = 0.0f32;
    for (_, prob) in candidates.iter_mut() {
        *prob = (*prob - max).exp();
        sum += *prob;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return candidates[0].0 as u32;
    }
    let inv_sum = 1.0 / sum;
    for (_, prob) in candidates.iter_mut() {
        *prob *= inv_sum;
    }

    let mut cutoff = candidates.len();
    if top_p < 1.0 {
        let mut cumsum = 0.0f32;
        for (i, &(_, prob)) in candidates.iter().enumerate() {
            cumsum += prob;
            if cumsum > top_p {
                cutoff = i + 1;
                break;
            }
        }

        let kept_sum: f32 = candidates[..cutoff].iter().map(|&(_, prob)| prob).sum();
        if kept_sum > 0.0 {
            let inv = 1.0 / kept_sum;
            for (_, prob) in candidates[..cutoff].iter_mut() {
                *prob *= inv;
            }
        }
    }

    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for &(idx, prob) in &candidates[..cutoff] {
        cumsum += prob;
        if cumsum > r {
            return idx as u32;
        }
    }

    candidates[cutoff - 1].0 as u32
}

fn bubble_up_last(candidates: &mut [(usize, f32)]) {
    let mut i = candidates.len() - 1;
    while i > 0 && candidates[i].1.total_cmp(&candidates[i - 1].1).is_gt() {
        candidates.swap(i, i - 1);
        i -= 1;
    }
}

fn argmax_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{Rng, SamplerConfig, sample};

    #[test]
    fn top_k_1_only_keeps_single_best_token() {
        let config = SamplerConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            repeat_penalty: 1.0,
        };
        let mut rng = Rng::new(42);
        for _ in 0..64 {
            let mut logits = vec![1.0, 10.0, 9.0];
            let token = sample(&mut logits, &config, &mut rng, &[]);
            assert_eq!(token, 1);
        }
    }

    #[test]
    fn empty_logits_returns_zero_token() {
        let config = SamplerConfig::default();
        let mut rng = Rng::new(7);
        let mut logits = Vec::new();
        let token = sample(&mut logits, &config, &mut rng, &[]);
        assert_eq!(token, 0);
    }

    #[test]
    fn top_k_top_p_keeps_minimal_candidate_set() {
        let config = SamplerConfig {
            temperature: 1.0,
            top_p: 0.6,
            top_k: 2,
            repeat_penalty: 1.0,
        };
        let mut rng = Rng::new(42);
        for _ in 0..64 {
            let mut logits = vec![10.0, 9.0, 0.0, -1.0];
            let token = sample(&mut logits, &config, &mut rng, &[]);
            assert_eq!(token, 0);
        }
    }
}
