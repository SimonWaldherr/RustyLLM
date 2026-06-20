---
name: german-technical-writing
description: Use this skill when the user asks in German for technical explanations, documentation, README text, release notes, CLI help text, commit or PR descriptions, architecture summaries, or precise editing of German developer-facing prose.
---

# German Technical Writing

## Style

- Use clear, direct German with technical terms where they are normal for developers.
- Prefer active voice and concrete nouns over abstract filler.
- Keep sentences short when explaining commands, flags, errors, or implementation behavior.
- Preserve code identifiers, CLI flags, API paths, filenames, and English library names exactly.

## Structure

1. State the key point first.
2. Explain the relevant mechanism or tradeoff.
3. Give commands, examples, or next steps only when they help the user act.

## Output Rules

- Do not translate identifiers such as `--skills-dir`, `SKILL.md`, `GenerationOptions`, or `/v1/chat/completions`.
- Use German punctuation and capitalization.
- Avoid marketing tone, exaggerated claims, and vague words like "einfach", unless the simplicity is the actual point.
- When editing existing prose, keep the original intent and shorten where possible.
