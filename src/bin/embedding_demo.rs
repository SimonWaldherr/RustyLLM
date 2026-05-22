use rusty_llm::runtime::{Runner, cosine_similarity};

/// Prints usage for the embedding demo utility.
fn usage(bin: &str) {
    eprintln!(
        "Usage: {bin} <model.gguf> [text-a] [text-b] [text-c]\n\
         Example:\n\
           {bin} ./models/embed.gguf \"Albert Einstein was a physicist\" \"Einstein developed relativity\" \"Bananas are yellow\""
    );
}

/// runs the CLI and prints fatal errors.
fn main() -> Result<(), String> {
    let mut args = std::env::args();
    let bin = args
        .next()
        .unwrap_or_else(|| String::from("embedding_demo"));
    let Some(model_path) = args.next() else {
        usage(&bin);
        return Err(String::from("missing model path"));
    };

    let text_a = args
        .next()
        .unwrap_or_else(|| String::from("Albert Einstein was a physicist."));
    let text_b = args
        .next()
        .unwrap_or_else(|| String::from("Einstein developed the theory of relativity."));
    let text_c = args
        .next()
        .unwrap_or_else(|| String::from("A banana is a tropical fruit."));

    let (runner, load) = Runner::from_path(&model_path)?;
    eprintln!(
        "Loaded model in {:.2}s ({} bytes)",
        load.load_time.as_secs_f32(),
        load.file_size_bytes
    );
    eprintln!("Architecture: {}", runner.architecture());
    if let Some(name) = runner.model_name() {
        eprintln!("Model: {}", name);
    }

    let emb_a = runner.embed(&text_a)?;
    let emb_b = runner.embed(&text_b)?;
    let emb_c = runner.embed(&text_c)?;

    let sim_ab = cosine_similarity(&emb_a.embedding, &emb_b.embedding)?;
    let sim_ac = cosine_similarity(&emb_a.embedding, &emb_c.embedding)?;
    let sim_bc = cosine_similarity(&emb_b.embedding, &emb_c.embedding)?;

    println!("dim={}", emb_a.embedding.len());
    println!(
        "tokens: a={} b={} c={}",
        emb_a.token_count, emb_b.token_count, emb_c.token_count
    );
    println!("cos(a,b)={:.6}", sim_ab);
    println!("cos(a,c)={:.6}", sim_ac);
    println!("cos(b,c)={:.6}", sim_bc);
    println!();
    println!("a: {}", text_a);
    println!("b: {}", text_b);
    println!("c: {}", text_c);

    Ok(())
}
