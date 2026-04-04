// main.rs — rusty-llm: dependency-free LLM inference
//
// Usage:
//   rusty-llm model.gguf --prompt "Hello" --max-tokens 256 --temp 0.7
//   echo "Hello" | rusty-llm model.gguf
//
// Supports: LLaMA, Mistral, Qwen2 and compatible architectures (GGUF format)
// Quantization: f32, f16, Q8_0, Q4_0
// SIMD: NEON (Apple Silicon), AVX2+FMA (x86_64), scalar fallback

mod gguf;
mod mmap;
mod model;
mod sampling;
mod simd;
mod tokenizer;

use std::env;
use std::io::{self, BufRead, Write};
use std::time::Instant;

fn architecture_supported(arch: &str) -> bool {
    matches!(arch, "llama" | "qwen2" | "gpt-oss")
}

fn print_usage(name: &str) {
    eprintln!("rusty-llm v0.1.0 — dependency-free LLM inference");
    eprintln!();
    eprintln!("Usage: {} <model.gguf> [options]", name);
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --prompt <text>       Input prompt (interactive if omitted)");
    eprintln!("  --max-tokens <N>      Max tokens to generate (default: 256)");
    eprintln!("  --temp <F>            Temperature (default: 0.7, 0=greedy)");
    eprintln!("  --top-p <F>           Nucleus sampling threshold (default: 0.9)");
    eprintln!("  --top-k <N>           Top-K filtering (default: 40)");
    eprintln!("  --repeat-penalty <F>  Repetition penalty (default: 1.1)");
    eprintln!("  --seed <N>            RNG seed (default: time-based)");
    eprintln!("  --threads <N>         Override thread count");
    eprintln!("  --system-prompt <T>   Override the default system prompt for chat templates");
}

fn print_system_info() {
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    eprintln!("System: {} threads", n_threads);

    #[cfg(target_arch = "aarch64")]
    eprintln!("SIMD: ARM NEON (native)");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            eprintln!("SIMD: AVX2 + FMA");
        } else if is_x86_feature_detected!("sse4.1") {
            eprintln!("SIMD: SSE4.1 (AVX2 not available)");
        } else {
            eprintln!("SIMD: scalar fallback");
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    eprintln!("SIMD: scalar fallback");
}

fn render_gpt_oss_prompt(tok: &tokenizer::Tokenizer, prompt: &str) -> Vec<u32> {
    let start = tok.special_id("<|start|>").unwrap_or(200006);
    let channel = tok.special_id("<|channel|>").unwrap_or(200005);
    let message = tok.special_id("<|message|>").unwrap_or(200008);
    let end = tok.special_id("<|end|>").unwrap_or(200007);
    let user = tok.special_id("user").unwrap_or_else(|| tok.encode_without_bos("user")[0]);
    let assistant = tok.special_id("assistant").unwrap_or_else(|| tok.encode_without_bos("assistant")[0]);
    let system = tok.special_id("system").unwrap_or_else(|| tok.encode_without_bos("system")[0]);
    let final_tok = tok.special_id("final").unwrap_or_else(|| tok.encode_without_bos("final")[0]);

    let mut tokens = Vec::new();
    tokens.push(start);
    tokens.push(system);
    tokens.push(message);
    tokens.extend(tok.encode_without_bos("You are ChatGPT, a helpful assistant."));
    tokens.push(end);
    tokens.push(start);
    tokens.push(user);
    tokens.push(message);
    tokens.extend(tok.encode_without_bos(prompt));
    tokens.push(end);
    tokens.push(start);
    tokens.push(assistant);
    tokens.push(channel);
    tokens.push(final_tok);
    tokens.push(message);
    tokens
}

fn render_header_chat_prompt(
    tok: &tokenizer::Tokenizer,
    system_prompt: &str,
    prompt: &str,
) -> Option<Vec<u32>> {
    let bot = tok.special_id("<|begin_of_text|>")?;
    let start_header = tok.special_id("<|start_header_id|>")?;
    let end_header = tok.special_id("<|end_header_id|>")?;
    let eot = tok.special_id("<|eot_id|>")?;
    let system = tok.special_id("system")?;
    let user = tok.special_id("user")?;
    let assistant = tok.special_id("assistant")?;

    let mut tokens = Vec::new();
    let push_header = |role_id: u32, out: &mut Vec<u32>| {
        out.push(start_header);
        out.push(role_id);
        out.push(end_header);
        out.extend(tok.encode_without_bos("\n\n"));
    };

    tokens.push(bot);
    push_header(system, &mut tokens);
    tokens.extend(tok.encode_without_bos(system_prompt));
    tokens.push(eot);

    push_header(user, &mut tokens);
    tokens.extend(tok.encode_without_bos(prompt));
    tokens.push(eot);

    push_header(assistant, &mut tokens);
    Some(tokens)
}

fn chat_template_kind(gguf: &gguf::GGUFFile) -> Option<&'static str> {
    let template = gguf.metadata.get("tokenizer.chat_template")?.as_str()?;
    if template.contains("<|start_header_id|>") && template.contains("<|eot_id|>") {
        Some("header-chat")
    } else {
        None
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        print_usage(&args[0]);
        std::process::exit(if args.len() < 2 { 1 } else { 0 });
    }

    let model_path = &args[1];

    // Parse CLI args
    let mut prompt = String::new();
    let mut max_tokens: usize = 256;
    let mut sampler_cfg = sampling::SamplerConfig::default();
    let mut seed: u64 = 0;
    let mut list_tensors = false;
    let mut threads_override: Option<usize> = None;
    let mut system_prompt = String::from("You are a helpful assistant.");

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" | "-p" => { i += 1; prompt = args[i].clone(); }
            "--max-tokens" | "-n" => { i += 1; max_tokens = args[i].parse().expect("Invalid --max-tokens"); }
            "--temp" | "-t" => { i += 1; sampler_cfg.temperature = args[i].parse().expect("Invalid --temp"); }
            "--top-p" => { i += 1; sampler_cfg.top_p = args[i].parse().expect("Invalid --top-p"); }
            "--top-k" => { i += 1; sampler_cfg.top_k = args[i].parse().expect("Invalid --top-k"); }
            "--repeat-penalty" => { i += 1; sampler_cfg.repeat_penalty = args[i].parse().expect("Invalid --repeat-penalty"); }
            "--seed" => { i += 1; seed = args[i].parse().expect("Invalid --seed"); }
            "--threads" => { i += 1; threads_override = Some(args[i].parse().expect("Invalid --threads")); }
            "--system-prompt" => { i += 1; system_prompt = args[i].clone(); }
            "--list-tensors" => { list_tensors = true; }
            other => {
                eprintln!("Unknown option: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    print_system_info();
    if let Some(n) = threads_override {
        simd::set_num_threads(n);
        eprintln!("Worker threads: {}", n.max(1));
    }

    // ── Load model via mmap ──
    eprintln!("\nLoading: {}", model_path);
    let t0 = Instant::now();

    let mmap = mmap::MmapFile::open(model_path).unwrap_or_else(|e| {
        eprintln!("Failed to open model: {}", e);
        std::process::exit(1);
    });

    let file_mb = mmap.len() as f64 / (1024.0 * 1024.0);
    eprintln!("File size: {:.1} MB (mmap'd)", file_mb);

    let gguf = gguf::GGUFFile::parse(mmap.as_slice()).unwrap_or_else(|e| {
        eprintln!("GGUF parse error: {}", e);
        std::process::exit(1);
    });

    if let Some(name) = gguf.get_str("general.name") {
        eprintln!("Model: {}", name);
    }
    if let Some(arch) = gguf.get_str("general.architecture") {
        eprintln!("Architecture: {}", arch);
        if list_tensors {
            for t in &gguf.tensors {
                eprintln!("{} {:?} {:?}", t.name, t.dtype, t.dims);
            }
            std::process::exit(0);
        }
        if !architecture_supported(arch) {
            eprintln!(
                "Unsupported architecture: {}. This runner currently supports llama, qwen2 and gpt-oss GGUF models only.",
                arch
            );
            eprintln!(
                "This model likely also requires additional tensor layouts/ops not implemented here (for example gpt-oss blocks and MXFP4 weights)."
            );
            std::process::exit(1);
        }
    }

    let arch = gguf.get_str("general.architecture").unwrap_or("llama").to_string();

    let tok = tokenizer::Tokenizer::from_metadata(&gguf.metadata);
    eprintln!("Tokenizer: {} tokens, BOS={}, EOS={}", tok.vocab_size(), tok.bos_id, tok.eos_id);

    let (config, standard_weights, gpt_oss_weights) = match arch.as_str() {
        "gpt-oss" => {
            let (config, weights) = model::load_gpt_oss_model(mmap.as_slice(), &gguf);
            (config, None, Some(weights))
        }
        _ => {
            let (config, weights) = model::load_model(mmap.as_slice(), &gguf);
            (config, Some(weights), None)
        }
    };
    let load_time = t0.elapsed();
    eprintln!("Loaded in {:.2}s ({:.0} MB/s)\n",
        load_time.as_secs_f32(),
        file_mb / load_time.as_secs_f64());

    // ── Interactive or single-shot ──
    if prompt.is_empty() {
        // Check for piped stdin
        if atty_is_stdin() {
            eprint!(">>> ");
            io::stderr().flush().unwrap();
            io::stdin().lock().read_line(&mut prompt).unwrap();
            prompt = prompt.trim().to_string();
        } else {
            // Read all of stdin
            let mut buf = String::new();
            io::stdin().lock().read_to_string_linewise(&mut buf);
            prompt = buf.trim().to_string();
        }
    }

    if prompt.is_empty() {
        eprintln!("No prompt provided.");
        std::process::exit(1);
    }

    // ── Tokenize ──
    let tokens = if arch == "gpt-oss" {
        render_gpt_oss_prompt(&tok, &prompt)
    } else if matches!(chat_template_kind(&gguf), Some("header-chat")) {
        render_header_chat_prompt(&tok, &system_prompt, &prompt).unwrap_or_else(|| tok.encode(&prompt))
    } else {
        tok.encode(&prompt)
    };
    eprintln!("Prompt: {} tokens", tokens.len());

    // ── Inference ──
    // Avoid allocating a full long-context KV cache (e.g. 128k) for short one-shot runs.
    let cache_len = std::cmp::min(config.max_seq_len, tokens.len() + max_tokens + 1);
    let mut cache = model::KVCache::new(config.n_layers, config.kv_dim, cache_len);
    let mut rng = if seed == 0 {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        sampling::Rng::new(t)
    } else {
        sampling::Rng::new(seed)
    };

    // Prefill
    let t_prefill = Instant::now();
    let mut logits = vec![];
    for (pos, &tok_id) in tokens.iter().enumerate() {
        logits = match arch.as_str() {
            "gpt-oss" => model::forward_gpt_oss(&config, gpt_oss_weights.as_ref().unwrap(), &mut cache, tok_id, pos),
            _ => model::forward(&config, standard_weights.as_ref().unwrap(), &mut cache, tok_id, pos),
        };
    }
    let prefill_time = t_prefill.elapsed();
    eprintln!("Prefill: {:.2}s ({:.1} tok/s)",
        prefill_time.as_secs_f32(),
        tokens.len() as f32 / prefill_time.as_secs_f32());

    // Decode
    let t_decode = Instant::now();
    let mut generated = Vec::new();
    let mut pos = tokens.len();
    let mut recent: Vec<u32> = tokens.clone();

    for _ in 0..max_tokens {
        let token = sampling::sample(&mut logits, &sampler_cfg, &mut rng, &recent);

        let is_stop = if arch == "gpt-oss" {
            token == tok.eos_id || token == 200002 || token == 200007
        } else {
            token == tok.eos_id
        };

        if is_stop {
            break;
        }

        let text = tok.decode_token(token);
        print!("{}", text);
        io::stdout().flush().unwrap();

        generated.push(token);
        recent.push(token);
        if recent.len() > 64 { recent.remove(0); }

        logits = match arch.as_str() {
            "gpt-oss" => model::forward_gpt_oss(&config, gpt_oss_weights.as_ref().unwrap(), &mut cache, token, pos),
            _ => model::forward(&config, standard_weights.as_ref().unwrap(), &mut cache, token, pos),
        };
        pos += 1;

        if pos >= cache_len {
            eprintln!("\n[context length limit reached]");
            break;
        }
    }

    let decode_time = t_decode.elapsed();
    let n_gen = generated.len();

    eprintln!("\n\n─── Stats ───────────────────────────────");
    eprintln!("Generated: {} tokens", n_gen);
    eprintln!("Decode: {:.2}s ({:.2} tok/s)",
        decode_time.as_secs_f32(),
        n_gen as f32 / decode_time.as_secs_f32());
    eprintln!("Total: {:.2}s", t0.elapsed().as_secs_f32());
}

/// Check if stdin is a terminal (not piped)
fn atty_is_stdin() -> bool {
    // Use isatty via libc — available on both macOS and Linux without deps
    extern "C" { fn isatty(fd: i32) -> i32; }
    unsafe { isatty(0) != 0 }
}

/// Helper to read all lines from stdin
trait ReadAllLines {
    fn read_to_string_linewise(&mut self, buf: &mut String);
}

impl<T: BufRead> ReadAllLines for T {
    fn read_to_string_linewise(&mut self, buf: &mut String) {
        loop {
            let mut line = String::new();
            match self.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => buf.push_str(&line),
                Err(_) => break,
            }
        }
    }
}
