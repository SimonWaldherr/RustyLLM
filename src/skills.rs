use crate::runtime::{ChatMessage, ChatRole, SkillConfig};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

const SUMMARY_LINE_LIMIT: usize = 120;
const SUMMARY_BYTE_LIMIT: usize = 8 * 1024;
const MAX_DISCOVERY_DEPTH: usize = 6;
const MAX_DISCOVERY_DIRS: usize = 2000;
const SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".next",
    "build",
    "dist",
    "node_modules",
    "target",
];

#[derive(Clone, Debug)]
pub struct SkillContextBundle {
    pub system_prompt_suffix: String,
    pub loaded_paths: Vec<String>,
    pub loaded_names: Vec<String>,
}

#[derive(Clone, Debug)]
struct SkillSummary {
    id: String,
    path: PathBuf,
    name: String,
    title: String,
    weights: HashMap<String, usize>,
}

/// Selects relevant skills for the current prompt and loads only their `SKILL.md` bodies.
pub fn prepare_skill_context(
    config: &SkillConfig,
    messages: &[ChatMessage],
    already_loaded: &HashSet<String>,
) -> Result<SkillContextBundle, String> {
    if !config.is_enabled() {
        return Ok(empty_bundle());
    }

    let directory = config
        .directory
        .as_deref()
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .ok_or_else(|| String::from("Skills directory is empty."))?;
    let root = Path::new(directory);
    if !root.is_dir() {
        return Err(format!(
            "Skills directory does not exist or is not a directory: {}",
            root.display()
        ));
    }

    let prompt = selection_text(messages);
    if prompt.trim().is_empty() {
        return Ok(empty_bundle());
    }

    let summaries = discover_skill_summaries(root)?;
    if summaries.is_empty() {
        return Ok(empty_bundle());
    }

    let selected = select_relevant_skills(&prompt, &summaries, already_loaded, config.max_skills);
    if selected.is_empty() {
        return Ok(empty_bundle());
    }

    let mut loaded_paths = Vec::new();
    let mut loaded_names = Vec::new();
    let mut blocks = Vec::new();
    for summary in selected {
        let content = load_skill_body(&summary.path, config.max_bytes_per_skill)?;
        loaded_paths.push(summary.id.clone());
        loaded_names.push(summary.name.clone());
        blocks.push(format!(
            "<skill name=\"{}\" path=\"{}\">\n{}\n</skill>",
            xml_escape_attr(&summary.name),
            xml_escape_attr(&summary.id),
            content.trim()
        ));
    }

    let system_prompt_suffix = format!(
        "Relevant skills loaded for this prompt. Apply these SKILL.md instructions when they are useful for the user's request; otherwise answer normally.\n\n{}",
        blocks.join("\n\n")
    );

    Ok(SkillContextBundle {
        system_prompt_suffix,
        loaded_paths,
        loaded_names,
    })
}

/// Appends selected skill instructions to the base system prompt.
pub fn append_skill_context(base_system_prompt: &str, bundle: &SkillContextBundle) -> String {
    if bundle.system_prompt_suffix.trim().is_empty() {
        return base_system_prompt.to_string();
    }
    if base_system_prompt.trim().is_empty() {
        return bundle.system_prompt_suffix.clone();
    }
    format!(
        "{}\n\n{}",
        base_system_prompt.trim_end(),
        bundle.system_prompt_suffix
    )
}

fn empty_bundle() -> SkillContextBundle {
    SkillContextBundle {
        system_prompt_suffix: String::new(),
        loaded_paths: Vec::new(),
        loaded_names: Vec::new(),
    }
}

fn selection_text(messages: &[ChatMessage]) -> String {
    if let Some(message) = messages
        .iter()
        .rev()
        .find(|message| matches!(message.role, ChatRole::User))
    {
        return message.content.clone();
    }
    messages
        .iter()
        .map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn discover_skill_summaries(root: &Path) -> Result<Vec<SkillSummary>, String> {
    let mut summaries = Vec::new();
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    let mut dirs_seen = 0usize;

    while let Some((dir, depth)) = stack.pop() {
        dirs_seen += 1;
        if dirs_seen > MAX_DISCOVERY_DIRS {
            return Err(format!(
                "Skills directory scan exceeded {} directories under {}.",
                MAX_DISCOVERY_DIRS,
                root.display()
            ));
        }
        let mut entries = fs::read_dir(&dir)
            .map_err(|err| format!("Failed to read skills directory {}: {}", dir.display(), err))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| {
                format!(
                    "Failed to read entry in skills directory {}: {}",
                    dir.display(),
                    err
                )
            })?;
        entries.sort_by_key(|entry| entry.path());
        for entry in entries {
            let path = entry.path();
            let file_type = entry.file_type().map_err(|err| {
                format!("Failed to inspect skills path {}: {}", path.display(), err)
            })?;
            if file_type.is_dir() {
                if depth < MAX_DISCOVERY_DEPTH && !should_skip_dir(&path) {
                    stack.push((path, depth + 1));
                }
            } else if path.file_name().and_then(|name| name.to_str()) == Some("SKILL.md") {
                if let Some(summary) = read_skill_summary(root, &path)? {
                    summaries.push(summary);
                }
            }
        }
    }

    summaries.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(summaries)
}

fn read_skill_summary(root: &Path, path: &Path) -> Result<Option<SkillSummary>, String> {
    let file = File::open(path)
        .map_err(|err| format!("Failed to open skill {}: {}", path.display(), err))?;
    let reader = BufReader::new(file);
    let mut title = String::new();
    let mut name = String::new();
    let mut description = String::new();
    let mut first_paragraph = String::new();
    let mut bytes = 0usize;

    for line in reader.lines().take(SUMMARY_LINE_LIMIT) {
        let line =
            line.map_err(|err| format!("Failed to read skill {}: {}", path.display(), err))?;
        bytes += line.len();
        if bytes > SUMMARY_BYTE_LIMIT {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed == "---" {
            continue;
        }

        if title.is_empty() {
            if let Some(stripped) = trimmed.strip_prefix("# ") {
                title = stripped.trim().to_string();
                continue;
            }
        }

        let lower = trimmed.to_ascii_lowercase();
        if let Some(value) = trimmed
            .split_once(':')
            .filter(|(key, _)| key.eq_ignore_ascii_case("name"))
            .map(|(_, value)| clean_frontmatter_value(value))
        {
            if name.is_empty() {
                name = value;
            }
            continue;
        }
        if let Some(value) = trimmed
            .split_once(':')
            .filter(|(key, _)| key.eq_ignore_ascii_case("description"))
            .map(|(_, value)| clean_frontmatter_value(value))
        {
            if description.is_empty() {
                description = value;
            }
            continue;
        }

        if first_paragraph.is_empty()
            && !trimmed.starts_with('#')
            && !lower.starts_with("name:")
            && !lower.starts_with("description:")
        {
            first_paragraph = trimmed.to_string();
        }
    }

    if title.is_empty() {
        title = if !name.is_empty() {
            name.clone()
        } else {
            path.parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                .unwrap_or("skill")
                .to_string()
        };
    }
    if description.is_empty() {
        description = first_paragraph;
    }
    if description.trim().is_empty() {
        return Ok(None);
    }
    if name.is_empty() {
        name = normalize_skill_name(&title);
    }

    let id = canonical_skill_id(path);
    let mut weights = HashMap::new();
    add_weighted_terms(&mut weights, &name, 5);
    add_weighted_terms(&mut weights, &title, 4);
    add_weighted_terms(&mut weights, &description, 2);
    if let Ok(relative) = path.strip_prefix(root) {
        add_weighted_terms(&mut weights, &relative.display().to_string(), 3);
    }

    Ok(Some(SkillSummary {
        id,
        path: path.to_path_buf(),
        name,
        title,
        weights,
    }))
}

fn should_skip_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| SKIP_DIRS.iter().any(|skip| name.eq_ignore_ascii_case(skip)))
        .unwrap_or(false)
}

fn select_relevant_skills<'a>(
    prompt: &str,
    summaries: &'a [SkillSummary],
    already_loaded: &HashSet<String>,
    max_skills: usize,
) -> Vec<&'a SkillSummary> {
    let prompt_lower = prompt.to_lowercase();
    let terms = tokenize(prompt);
    if terms.is_empty() {
        return Vec::new();
    }

    let mut scored = Vec::new();
    for summary in summaries {
        if already_loaded.contains(&summary.id) {
            continue;
        }
        let mut score = 0usize;
        for term in &terms {
            score += summary.weights.get(term).copied().unwrap_or(0);
        }

        let title_lower = summary.title.to_lowercase();
        if title_lower.len() >= 4 && prompt_lower.contains(&title_lower) {
            score += 8;
        }
        let name_lower = summary.name.to_lowercase();
        if name_lower.len() >= 4 && prompt_lower.contains(&name_lower) {
            score += 8;
        }
        if score >= 4 {
            scored.push((score, summary));
        }
    }

    scored.sort_by(|(left_score, left), (right_score, right)| {
        right_score
            .cmp(left_score)
            .then_with(|| left.id.cmp(&right.id))
    });
    scored
        .into_iter()
        .take(max_skills.max(1))
        .map(|(_, summary)| summary)
        .collect()
}

fn load_skill_body(path: &Path, max_bytes: usize) -> Result<String, String> {
    let content = fs::read_to_string(path)
        .map_err(|err| format!("Failed to load skill {}: {}", path.display(), err))?;
    Ok(truncate_utf8(content, max_bytes))
}

fn add_weighted_terms(weights: &mut HashMap<String, usize>, text: &str, weight: usize) {
    for term in tokenize(text) {
        *weights.entry(term).or_insert(0) += weight;
    }
}

fn tokenize(text: &str) -> HashSet<String> {
    let mut terms = HashSet::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            current.extend(ch.to_lowercase());
        } else {
            push_token(&mut terms, &mut current);
        }
    }
    push_token(&mut terms, &mut current);
    terms
}

fn push_token(terms: &mut HashSet<String>, current: &mut String) {
    if current.len() >= 3 && !is_stopword(current) {
        terms.insert(std::mem::take(current));
    } else {
        current.clear();
    }
}

fn is_stopword(term: &str) -> bool {
    matches!(
        term,
        "and"
            | "are"
            | "can"
            | "das"
            | "den"
            | "der"
            | "die"
            | "ein"
            | "eine"
            | "einen"
            | "einer"
            | "for"
            | "ist"
            | "mit"
            | "not"
            | "oder"
            | "prompt"
            | "skill"
            | "skills"
            | "the"
            | "und"
            | "use"
            | "using"
            | "was"
            | "wenn"
            | "wer"
            | "wie"
            | "with"
            | "you"
            | "zur"
    )
}

fn clean_frontmatter_value(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .to_string()
}

fn normalize_skill_name(value: &str) -> String {
    let mut out = String::new();
    let mut last_hyphen = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_hyphen = false;
        } else if !last_hyphen && !out.is_empty() {
            out.push('-');
            last_hyphen = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        String::from("skill")
    } else {
        out
    }
}

fn canonical_skill_id(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

fn truncate_utf8(mut text: String, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    text.push_str("\n\n[Skill truncated]");
    text
}

fn xml_escape_attr(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn temp_skill_root() -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "rusty-llm-skills-{}-{}-{}",
            std::process::id(),
            suffix,
            counter
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn selects_matching_skill_and_skips_loaded() {
        let root = temp_skill_root();
        let rust_dir = root.join("rust-review");
        fs::create_dir_all(&rust_dir).unwrap();
        fs::write(
            rust_dir.join("SKILL.md"),
            "---\nname: rust-review\ndescription: Review Rust ownership and borrow checker issues.\n---\n# Rust Review\nUse when reviewing Rust code.\n",
        )
        .unwrap();

        let config = SkillConfig {
            directory: Some(root.display().to_string()),
            max_skills: 3,
            max_bytes_per_skill: 4096,
        };
        let messages = [ChatMessage::user(
            "Bitte reviewe diesen Rust Borrow Checker Fehler.",
        )];
        let loaded = HashSet::new();
        let bundle = prepare_skill_context(&config, &messages, &loaded).unwrap();
        assert_eq!(bundle.loaded_names, vec![String::from("rust-review")]);
        assert!(
            bundle
                .system_prompt_suffix
                .contains("Review Rust ownership")
        );

        let loaded = bundle.loaded_paths.iter().cloned().collect();
        let bundle = prepare_skill_context(&config, &messages, &loaded).unwrap();
        assert!(bundle.loaded_paths.is_empty());
        assert!(bundle.system_prompt_suffix.is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn ignores_unrelated_skills() {
        let root = temp_skill_root();
        let sql_dir = root.join("sql");
        fs::create_dir_all(&sql_dir).unwrap();
        fs::write(
            sql_dir.join("SKILL.md"),
            "---\nname: sql\ndescription: Optimize database indexes and SQL queries.\n---\n",
        )
        .unwrap();

        let config = SkillConfig {
            directory: Some(root.display().to_string()),
            max_skills: 3,
            max_bytes_per_skill: 4096,
        };
        let messages = [ChatMessage::user("Schreibe ein Haiku über Schnee.")];
        let bundle = prepare_skill_context(&config, &messages, &HashSet::new()).unwrap();
        assert!(bundle.loaded_paths.is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn bundled_default_skill_authoring_triggers() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("skills/default");
        let config = SkillConfig {
            directory: Some(root.display().to_string()),
            max_skills: 3,
            max_bytes_per_skill: 4096,
        };
        let messages = [ChatMessage::user(
            "Recherchiere best practices fuer Skills und optimiere SKILL.md Dateien.",
        )];
        let bundle = prepare_skill_context(&config, &messages, &HashSet::new()).unwrap();
        assert!(
            bundle
                .loaded_names
                .contains(&String::from("skill-authoring"))
        );
        assert!(
            bundle
                .system_prompt_suffix
                .contains("Description Checklist")
        );
    }
}
