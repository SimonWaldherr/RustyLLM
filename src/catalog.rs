use crate::gguf::GGUFFile;
use crate::runtime::architecture_supported;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const LM_STUDIO_COMMUNITY_SUBDIR: &str = ".cache/lm-studio/models/lmstudio-community";

#[derive(Clone, Debug)]
pub struct ModelEntry {
    pub id: String,
    pub repository: String,
    pub file_name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub architecture: Option<String>,
    pub model_name: Option<String>,
    pub is_projector: bool,
    pub is_supported: bool,
}

impl ModelEntry {
    pub fn status(&self) -> &'static str {
        if self.is_projector {
            "projector"
        } else if self.is_supported {
            "supported"
        } else {
            "unsupported"
        }
    }
}

pub fn default_model_dir() -> PathBuf {
    if let Ok(path) = env::var("RUSTY_LLM_MODEL_DIR") {
        if !path.trim().is_empty() {
            return PathBuf::from(path);
        }
    }

    if let Ok(home) = env::var("HOME") {
        return PathBuf::from(home).join(LM_STUDIO_COMMUNITY_SUBDIR);
    }

    PathBuf::from(LM_STUDIO_COMMUNITY_SUBDIR)
}

pub fn discover_models(root: &Path) -> Result<Vec<ModelEntry>, String> {
    let mut files = Vec::new();
    collect_gguf_files(root, &mut files)
        .map_err(|err| format!("Failed to scan {}: {}", root.display(), err))?;
    files.sort();

    let mut entries = Vec::new();
    for path in files {
        match inspect_model(root, &path) {
            Ok(entry) => entries.push(entry),
            Err(err) => eprintln!("Skipping {}: {}", path.display(), err),
        }
    }

    entries.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(entries)
}

pub fn resolve_model_path(selection: Option<&str>, model_dir: &Path) -> Result<PathBuf, String> {
    if let Some(selection) = selection {
        let selected_path = Path::new(selection);
        if selected_path.exists() {
            if selected_path.is_file() {
                return Ok(selected_path.to_path_buf());
            }
            if selected_path.is_dir() {
                return choose_from_directory(selected_path, None);
            }
            return Err(format!(
                "Model path is neither a file nor a directory: {}",
                selection
            ));
        }

        let entries = discover_models(model_dir)?;
        return select_model(&entries, selection).map(|entry| entry.path.clone());
    }

    choose_from_directory(model_dir, None)
}

pub fn select_model<'a>(
    entries: &'a [ModelEntry],
    selector: &str,
) -> Result<&'a ModelEntry, String> {
    let selector = selector.trim();
    if selector.is_empty() {
        return Err(String::from("Model selector must not be empty."));
    }

    let usable: Vec<&ModelEntry> = entries
        .iter()
        .filter(|entry| entry.is_supported && !entry.is_projector)
        .collect();
    let matches = matching_entries(&usable, selector);
    if matches.len() == 1 {
        return Ok(matches[0]);
    }
    if matches.len() > 1 {
        return Err(format_ambiguous(selector, &matches));
    }

    let all: Vec<&ModelEntry> = entries.iter().collect();
    let unsupported_matches = matching_entries(&all, selector);
    if unsupported_matches.len() == 1 {
        let entry = unsupported_matches[0];
        return Err(format!(
            "Model '{}' matched {}, but it is marked as {} (architecture: {}).",
            selector,
            entry.id,
            entry.status(),
            entry.architecture.as_deref().unwrap_or("unknown")
        ));
    }
    if unsupported_matches.len() > 1 {
        return Err(format_ambiguous(selector, &unsupported_matches));
    }

    Err(format!(
        "No GGUF model matched '{}'. Use --list-models to see available models in {}.",
        selector,
        entries
            .first()
            .and_then(|entry| entry.path.parent())
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| String::from("the model directory"))
    ))
}

pub fn print_model_list(entries: &[ModelEntry]) {
    if entries.is_empty() {
        println!("No GGUF files found.");
        return;
    }

    println!("{:<62} {:<14} {:<12} size", "id", "architecture", "status");
    for entry in entries {
        let size_gb = entry.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        println!(
            "{:<62} {:<14} {:<12} {:.2} GB",
            truncate(&entry.id, 62),
            truncate(entry.architecture.as_deref().unwrap_or("unknown"), 14),
            entry.status(),
            size_gb
        );
    }
}

fn choose_from_directory(dir: &Path, selector: Option<&str>) -> Result<PathBuf, String> {
    let entries = discover_models(dir)?;
    if let Some(selector) = selector {
        return select_model(&entries, selector).map(|entry| entry.path.clone());
    }

    let usable: Vec<&ModelEntry> = entries
        .iter()
        .filter(|entry| entry.is_supported && !entry.is_projector)
        .collect();

    match usable.len() {
        0 => Err(format!(
            "No supported text GGUF models found in {}.",
            dir.display()
        )),
        1 => Ok(usable[0].path.clone()),
        _ => Err(format!(
            "Found multiple GGUF models in {}. Choose one with --model <name> or pass an exact .gguf path.\n\n{}",
            dir.display(),
            format_model_choices(&usable)
        )),
    }
}

fn collect_gguf_files(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_gguf_files(&path, out)?;
        } else if file_type.is_file()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
        {
            out.push(path);
        }
    }
    Ok(())
}

fn inspect_model(root: &Path, path: &Path) -> Result<ModelEntry, String> {
    let metadata = fs::metadata(path).map_err(|err| err.to_string())?;
    let mmap = crate::mmap::MmapFile::open(
        path.to_str()
            .ok_or_else(|| format!("Non-UTF-8 model path: {}", path.display()))?,
    )
    .map_err(|err| err.to_string())?;
    let gguf = GGUFFile::parse_quiet(mmap.as_slice())?;

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.gguf")
        .to_string();
    let repository = path
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_string();
    let id = path
        .strip_prefix(root)
        .ok()
        .and_then(|rel| rel.to_str())
        .map(|rel| rel.trim_end_matches(".gguf").to_string())
        .unwrap_or_else(|| file_name.trim_end_matches(".gguf").to_string());
    let architecture = gguf
        .get_str("general.architecture")
        .map(|arch| arch.to_string());
    let model_name = gguf.get_str("general.name").map(|name| name.to_string());
    let lowered = file_name.to_ascii_lowercase();
    let is_projector = lowered.starts_with("mmproj-")
        || lowered.contains("mmproj")
        || architecture
            .as_deref()
            .map(|arch| arch.eq_ignore_ascii_case("clip"))
            .unwrap_or(false);
    let is_supported = architecture
        .as_deref()
        .map(architecture_supported)
        .unwrap_or(false);

    Ok(ModelEntry {
        id,
        repository,
        file_name,
        path: path.to_path_buf(),
        size_bytes: metadata.len(),
        architecture,
        model_name,
        is_projector,
        is_supported,
    })
}

fn matching_entries<'a>(entries: &[&'a ModelEntry], selector: &str) -> Vec<&'a ModelEntry> {
    let needle = selector.to_ascii_lowercase();
    let mut exact = Vec::new();
    let mut partial = Vec::new();

    for entry in entries {
        let keys = [
            entry.id.as_str(),
            entry.repository.as_str(),
            entry.file_name.as_str(),
            entry.model_name.as_deref().unwrap_or(""),
            entry.path.to_str().unwrap_or(""),
        ];

        if keys.iter().any(|key| key.eq_ignore_ascii_case(selector)) {
            exact.push(*entry);
        } else if keys
            .iter()
            .any(|key| key.to_ascii_lowercase().contains(&needle))
        {
            partial.push(*entry);
        }
    }

    if exact.is_empty() { partial } else { exact }
}

fn format_ambiguous(selector: &str, entries: &[&ModelEntry]) -> String {
    format!(
        "Model selector '{}' matched multiple GGUF files:\n\n{}",
        selector,
        format_model_choices(entries)
    )
}

fn format_model_choices(entries: &[&ModelEntry]) -> String {
    entries
        .iter()
        .map(|entry| {
            format!(
                "  - {} [{}; {}]",
                entry.id,
                entry.architecture.as_deref().unwrap_or("unknown"),
                entry.status()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }

    let keep = max_chars.saturating_sub(1);
    let mut out: String = value.chars().take(keep).collect();
    out.push('~');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, arch: &str, is_projector: bool) -> ModelEntry {
        let file_name = id
            .rsplit('/')
            .next()
            .map(|name| format!("{}.gguf", name))
            .unwrap_or_else(|| String::from("model.gguf"));
        let repository = id.split('/').next().unwrap_or("repo").to_string();
        ModelEntry {
            id: id.to_string(),
            repository,
            file_name,
            path: PathBuf::from(format!("/models/{}.gguf", id)),
            size_bytes: 1024,
            architecture: Some(arch.to_string()),
            model_name: None,
            is_projector,
            is_supported: architecture_supported(arch),
        }
    }

    #[test]
    fn select_model_ignores_projector_matches_when_text_model_exists() {
        let entries = vec![
            entry("gemma-4/mmproj-gemma-4", "clip", true),
            entry("gemma-4/gemma-4-Q4_K_M", "gemma4", false),
        ];

        let selected = select_model(&entries, "gemma-4").unwrap();

        assert_eq!(selected.id, "gemma-4/gemma-4-Q4_K_M");
    }

    #[test]
    fn select_model_reports_ambiguous_text_matches() {
        let entries = vec![
            entry("phi-4/phi-4-Q4_K_M", "phi3", false),
            entry("Phi-3.1-mini/Phi-3.1-mini-Q4_K_M", "phi3", false),
        ];

        let err = select_model(&entries, "phi").unwrap_err();

        assert!(err.contains("matched multiple"));
    }
}
