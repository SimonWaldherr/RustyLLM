use std::collections::HashMap;
use std::env;
use std::fs;

// Import from the library
use rusty_llm::gguf::GGUFFile;

/// runs the CLI and prints fatal errors.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <gguf_file>", args[0]);
        std::process::exit(1);
    }

    let gguf_path = &args[1];

    // Read the file into memory
    let data = fs::read(gguf_path)?;
    let gguf = GGUFFile::parse(&data)?;

    println!("\n=== GEMMA-4 GGUF Layer Architecture Analysis ===\n");

    // Organize tensors by block
    let mut blocks: HashMap<String, Vec<_>> = HashMap::new();

    for tensor in &gguf.tensors {
        // Extract block number from tensor name (e.g., "blk.5.attn_q" -> "blk.5")
        if let Some(blk_start) = tensor.name.find("blk.") {
            if let Some(blk_end) = tensor.name[blk_start..].find('.') {
                let block_key = &tensor.name[blk_start..blk_start + blk_end];
                blocks
                    .entry(block_key.to_string())
                    .or_insert_with(Vec::new)
                    .push(tensor);
            }
        }
    }

    // Sort blocks for consistent output
    let mut block_keys: Vec<_> = blocks.keys().collect();
    block_keys.sort_by_key(|b| {
        b.strip_prefix("blk.")
            .and_then(|n| n.parse::<u32>().ok())
            .unwrap_or(999)
    });

    // Analyze specific layers: normal layers (0, 1, 2, 3, 4, 6, 7...) vs special layers (5, 11, 17, 23, 29)
    let special_layers = [5, 11, 17, 23, 29];
    let analysis_layers = vec![0, 5, 11];

    for layer_num in analysis_layers {
        let block_key = format!("blk.{}", layer_num);
        if let Some(tensors) = blocks.get(&block_key) {
            let is_special = special_layers.contains(&layer_num);
            println!(
                "{}=== {} (Layer {}) {} ===",
                if layer_num == 0 { "" } else { "\n" },
                block_key,
                layer_num,
                if is_special {
                    "[SPECIAL LAYER]"
                } else {
                    "[NORMAL LAYER]"
                }
            );
            println!("Tensor Count: {}\n", tensors.len());

            // Categorize tensors by component
            let mut categories: HashMap<String, Vec<_>> = HashMap::new();
            for tensor in tensors {
                let component = if tensor.name.contains("attn") {
                    "ATTENTION"
                } else if tensor.name.contains("ffn") || tensor.name.contains("moe") {
                    "FEED_FORWARD / MoE"
                } else if tensor.name.contains("norm") {
                    "NORMALIZATION"
                } else {
                    "OTHER"
                };
                categories
                    .entry(component.to_string())
                    .or_insert_with(Vec::new)
                    .push(tensor);
            }

            // Print by category
            let mut cat_keys: Vec<_> = categories.keys().collect();
            cat_keys.sort();

            for category in cat_keys {
                let tensors = &categories[category];
                println!("  [{}]", category);
                for tensor in tensors {
                    let dtype_name = format!("{:?}", tensor.dtype);
                    let shape = tensor
                        .dims
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(" x ");
                    let short_name = tensor
                        .name
                        .strip_prefix(&format!("{}.", block_key))
                        .unwrap_or(&tensor.name);
                    println!(
                        "    {:40} | shape: {:20} | dtype: {}",
                        short_name, shape, dtype_name
                    );
                }
                println!();
            }
        }
    }

    // Now do a detailed comparison
    println!("\n=== STRUCTURAL COMPARISON: blk.0 (Normal) vs blk.5 (Special) ===\n");

    if let (Some(blk0), Some(blk5)) = (blocks.get("blk.0"), blocks.get("blk.5")) {
        compare_blocks(blk0, blk5, "blk.0", "blk.5");
    }

    // Additional comparisons
    println!("\n=== STRUCTURAL COMPARISON: blk.0 vs blk.11 ===\n");
    if let (Some(blk0), Some(blk11)) = (blocks.get("blk.0"), blocks.get("blk.11")) {
        compare_blocks(blk0, blk11, "blk.0", "blk.11");
    }

    Ok(())
}

/// Compares selected tensor blocks for the GGUF analysis utility.
fn compare_blocks(
    blk_a: &[&rusty_llm::gguf::TensorInfo],
    blk_b: &[&rusty_llm::gguf::TensorInfo],
    name_a: &str,
    name_b: &str,
) {
    // Extract just the tensor suffixes (after the block name)
    let mut map_a: HashMap<String, _> = HashMap::new();
    let mut map_b: HashMap<String, _> = HashMap::new();

    for tensor in blk_a {
        if let Some(suffix) = tensor.name.strip_prefix(&format!("{}.", name_a)) {
            map_a.insert(suffix.to_string(), tensor);
        }
    }

    for tensor in blk_b {
        if let Some(suffix) = tensor.name.strip_prefix(&format!("{}.", name_b)) {
            map_b.insert(suffix.to_string(), tensor);
        }
    }

    let mut all_keys: Vec<_> = map_a.keys().chain(map_b.keys()).collect();
    all_keys.sort();
    all_keys.dedup();

    let mut differences = Vec::new();

    for key in all_keys {
        match (map_a.get(key), map_b.get(key)) {
            (Some(t_a), Some(t_b)) => {
                let shape_a = format_dims(&t_a.dims);
                let shape_b = format_dims(&t_b.dims);
                let dtype_a = format!("{:?}", t_a.dtype);
                let dtype_b = format!("{:?}", t_b.dtype);

                if shape_a != shape_b || dtype_a != dtype_b {
                    differences.push(format!(
                        "{:40} | {}: {} ({}) | {}: {} ({})",
                        key, name_a, shape_a, dtype_a, name_b, shape_b, dtype_b
                    ));
                }
            }
            (Some(t), None) => {
                let shape = format_dims(&t.dims);
                let dtype = format!("{:?}", t.dtype);
                differences.push(format!(
                    "{:40} | {}: {} ({}) | {}: [MISSING]",
                    key, name_a, shape, dtype, name_b
                ));
            }
            (None, Some(t)) => {
                let shape = format_dims(&t.dims);
                let dtype = format!("{:?}", t.dtype);
                differences.push(format!(
                    "{:40} | {}: [MISSING] | {}: {} ({})",
                    key, name_a, name_b, shape, dtype
                ));
            }
            (None, None) => {} // shouldn't happen
        }
    }

    if differences.is_empty() {
        println!("✓ All tensors match between {} and {}!", name_a, name_b);
    } else {
        println!(
            "Found {} difference(s) between {} and {}:\n",
            differences.len(),
            name_a,
            name_b
        );
        for diff in &differences {
            println!("  {}", diff);
        }
    }

    // Summary statistics
    println!("\nSummary:");
    println!("  {} tensors:   {}", name_a, map_a.len());
    println!("  {} tensors:  {}", name_b, map_b.len());

    let only_in_a: Vec<_> = map_a.keys().filter(|k| !map_b.contains_key(*k)).collect();
    let only_in_b: Vec<_> = map_b.keys().filter(|k| !map_a.contains_key(*k)).collect();

    if !only_in_a.is_empty() {
        println!("  Only in {}: {:?}", name_a, only_in_a);
    }
    if !only_in_b.is_empty() {
        println!("  Only in {}: {:?}", name_b, only_in_b);
    }
}

/// Formats tensor dimensions for CLI display.
fn format_dims(dims: &[u64]) -> String {
    dims.iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(" x ")
}
