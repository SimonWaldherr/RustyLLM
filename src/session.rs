//! Persistent per-conversation KV-cache sessions.
//!
//! Each [`Session`] retains the full KV cache, scratch decode buffers, the
//! tokenised prompt/response history and the logits from the last forward
//! pass.  On every new request the new prompt is compared against the cached
//! token sequence with a Longest-Common-Prefix check: only the differing
//! suffix needs to be prefilled, giving a significant latency reduction for
//! multi-turn conversations.
//!
//! [`SessionStore`] manages a bounded set of sessions with LRU eviction.

use crate::model::{DecodeBuffer, KVCache};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ─── Session ─────────────────────────────────────────────────────────────────

/// All persistent state for a single conversation.
pub struct Session {
    /// Full KV cache, allocated once for the session's lifetime.
    pub kv_cache: KVCache,
    /// Scratch buffers reused across every forward pass in this session.
    pub decode_buf: DecodeBuffer,
    /// All tokens currently reflected in `kv_cache`
    /// (last prompt tokens + generated tokens from the previous turn).
    pub cached_tokens: Vec<u32>,
    /// Logits produced by the last forward pass; used as the starting point
    /// for the next sampling step, avoiding one redundant forward call.
    pub last_logits: Vec<f32>,
    /// Ring buffer of recent tokens fed to the repeat-penalty sampler.
    pub recent: VecDeque<u32>,
    /// Wall-clock time of the most recent access, used for LRU eviction.
    pub last_used: Instant,
    /// Cumulative count of prompt tokens served from the cache (no re-eval).
    pub cached_tokens_served: usize,
    /// Cumulative count of tokens that required a fresh forward pass.
    pub evaluated_tokens: usize,
}

impl Session {
    pub fn new(kv_cache: KVCache, decode_buf: DecodeBuffer) -> Self {
        Self {
            kv_cache,
            decode_buf,
            cached_tokens: Vec::new(),
            last_logits: Vec::new(),
            recent: VecDeque::new(),
            last_used: Instant::now(),
            cached_tokens_served: 0,
            evaluated_tokens: 0,
        }
    }

    /// Clear all cached state, keeping the allocated buffers so they can be
    /// reused for the next turn without re-allocating heap memory.
    pub fn reset(&mut self) {
        self.cached_tokens.clear();
        self.last_logits.clear();
        self.recent.clear();
        self.cached_tokens_served = 0;
        self.evaluated_tokens = 0;
    }
}

// ─── SessionStore ─────────────────────────────────────────────────────────────

/// Thread-safe map from `conversation_id` → [`Session`] with LRU eviction.
///
/// The outer [`Mutex`] guards the map structure itself.  Each entry is an
/// `Arc<Mutex<Session>>` so that the map lock can be released before the
/// (potentially long-running) generation call, while still preventing
/// concurrent access to the *same* session from two racing requests.
pub struct SessionStore {
    sessions: Mutex<HashMap<String, Arc<Mutex<Session>>>>,
    max_sessions: usize,
    max_cached_tokens: usize,
}

impl SessionStore {
    /// Create a new store.
    ///
    /// * `max_sessions` – maximum number of live sessions.  When the limit is
    ///   reached the least-recently-used session is evicted first.
    /// * `max_cached_tokens` – per-session KV-cache length cap (in tokens).
    pub fn new(max_sessions: usize, max_cached_tokens: usize) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            max_sessions: max_sessions.max(1),
            max_cached_tokens: max_cached_tokens.max(1),
        }
    }

    /// Maximum per-session KV-cache capacity (tokens).
    pub fn max_cached_tokens(&self) -> usize {
        self.max_cached_tokens
    }

    /// Return the existing session for `id`, or create a new one via `create`.
    ///
    /// When the store is at capacity the least-recently-used session is evicted
    /// before the new entry is inserted.
    pub fn get_or_create(&self, id: &str, create: impl FnOnce() -> Session) -> Arc<Mutex<Session>> {
        let mut map = self.sessions.lock().expect("session store lock poisoned");
        if let Some(arc) = map.get(id) {
            return Arc::clone(arc);
        }
        // Evict LRU session when at capacity.
        if map.len() >= self.max_sessions {
            let lru_key = map
                .iter()
                .min_by_key(|(_, arc)| {
                    arc.lock()
                        .map(|s| s.last_used)
                        .unwrap_or_else(|_| Instant::now())
                })
                .map(|(k, _)| k.clone());
            if let Some(key) = lru_key {
                map.remove(&key);
            }
        }
        let arc = Arc::new(Mutex::new(create()));
        map.insert(id.to_string(), Arc::clone(&arc));
        arc
    }

    /// Remove a session explicitly (e.g. client-requested cache invalidation).
    pub fn delete(&self, id: &str) {
        let mut map = self.sessions.lock().expect("session store lock poisoned");
        map.remove(id);
    }

    /// Number of active sessions.
    pub fn len(&self) -> usize {
        self.sessions.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// `true` when no sessions are currently held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Config, DecodeBuffer, KVCache};

    fn dummy_session() -> Session {
        // Minimal KV cache and decode buffer for testing session management
        // (not used for actual inference).
        let kv = KVCache::new(1, 4, 4, 8);
        let cfg = Config {
            arch: String::from("llama"),
            dim: 4,
            hidden_dim: 8,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            vocab_size: 32,
            max_seq_len: 8,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            head_dim: 4,
            kv_dim: 4,
            kv_mul: 1,
            value_dim: 4,
            sliding_window: 0,
            expert_count: 0,
            expert_used_count: 0,
            rope_scaling_factor: 1.0,
            rope_original_context_length: 0,
        };
        let buf = DecodeBuffer::new(&cfg, 4, 1, 4);
        Session::new(kv, buf)
    }

    #[test]
    fn session_reset_clears_state() {
        let mut s = dummy_session();
        s.cached_tokens = vec![1, 2, 3];
        s.last_logits = vec![0.1, 0.2];
        s.cached_tokens_served = 5;
        s.evaluated_tokens = 10;
        s.reset();
        assert!(s.cached_tokens.is_empty());
        assert!(s.last_logits.is_empty());
        assert_eq!(s.cached_tokens_served, 0);
        assert_eq!(s.evaluated_tokens, 0);
    }

    #[test]
    fn store_creates_new_session_on_miss() {
        let store = SessionStore::new(4, 64);
        let arc = store.get_or_create("conv1", dummy_session);
        let _guard = arc.lock().unwrap();
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn store_returns_existing_session_on_hit() {
        let store = SessionStore::new(4, 64);
        let a1 = store.get_or_create("conv1", dummy_session);
        let a2 = store.get_or_create("conv1", dummy_session);
        assert!(Arc::ptr_eq(&a1, &a2));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn store_evicts_lru_when_full() {
        let store = SessionStore::new(2, 64);
        // Insert first session and mark it as older.
        {
            let arc = store.get_or_create("old", dummy_session);
            let mut s = arc.lock().unwrap();
            s.last_used = Instant::now() - std::time::Duration::from_secs(100);
        }
        store.get_or_create("recent", dummy_session);
        // Store is now full (2/2).  Adding a third should evict "old".
        store.get_or_create("new", dummy_session);
        assert_eq!(store.len(), 2);
        // "old" must have been evicted; "recent" and "new" survive.
        {
            let map = store.sessions.lock().unwrap();
            assert!(!map.contains_key("old"), "LRU session was not evicted");
            assert!(map.contains_key("recent"));
            assert!(map.contains_key("new"));
        }
    }

    #[test]
    fn store_delete_removes_session() {
        let store = SessionStore::new(4, 64);
        store.get_or_create("conv1", dummy_session);
        assert_eq!(store.len(), 1);
        store.delete("conv1");
        assert_eq!(store.len(), 0);
    }
}
