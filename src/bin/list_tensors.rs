use rusty_llm::gguf::GGUFFile;
use std::env;
use std::fs;

/// runs the CLI and prints fatal errors.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <gguf_file>", args[0]);
        std::process::exit(1);
    }

    let gguf_path = &args[1];
    let data = fs::read(gguf_path)?;
    let gguf = GGUFFile::parse(&data)?;

    println!("Total tensors: {}\n", gguf.tensors.len());

    // Print first 30 tensor names
    for (i, tensor) in gguf.tensors.iter().take(30).enumerate() {
        println!(
            "{}: {} -> shape: {:?}, dtype: {:?}",
            i, tensor.name, tensor.dims, tensor.dtype
        );
    }

    println!("\n...\n");

    // Print some blk tensors
    for (i, tensor) in gguf.tensors.iter().enumerate() {
        if tensor.name.contains("blk.0") && i < 100 {
            println!("blk.0 tensor: {}", tensor.name);
        }
    }

    Ok(())
}
