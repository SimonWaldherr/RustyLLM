#![allow(
    clippy::collapsible_if,
    clippy::collapsible_match,
    clippy::explicit_counter_loop,
    clippy::manual_checked_ops,
    clippy::manual_is_multiple_of,
    clippy::unnecessary_unwrap,
    clippy::useless_conversion
)]

use rusty_llm::gguf::GGUFFile;
use std::env;
use std::fs;

/// runs the CLI and prints fatal errors.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <gguf_file> [--all]", args[0]);
        std::process::exit(1);
    }

    let gguf_path = &args[1];
    let print_all = args.iter().any(|arg| arg == "--all");
    let data = fs::read(gguf_path)?;
    let gguf = GGUFFile::parse(&data)?;

    println!("Total tensors: {}\n", gguf.tensors.len());

    let tensors: Box<dyn Iterator<Item = (usize, &rusty_llm::gguf::TensorInfo)>> = if print_all {
        Box::new(gguf.tensors.iter().enumerate())
    } else {
        Box::new(gguf.tensors.iter().take(30).enumerate())
    };
    for (i, tensor) in tensors {
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
