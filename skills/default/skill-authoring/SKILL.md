---
name: skill-authoring
description: Use this skill when creating, reviewing, or improving SKILL.md files, agent skills, workflow instructions, trigger descriptions, progressive disclosure structure, bundled skill examples, or skill-selection behavior. Also use it when the user asks for best practices for skills.
---

# Skill Authoring

## Principles

1. Put the trigger contract in frontmatter: `name` is short and stable; `description` says when to use the skill and includes realistic user-intent keywords.
2. Keep the body procedural, not encyclopedic. Include what the model is likely to miss: local conventions, edge cases, validation steps, and preferred defaults.
3. Use progressive disclosure. Keep `SKILL.md` small; reference extra files only when the runtime can load them on demand.
4. Prefer concrete checklists, gotchas, and output templates over generic advice.
5. Avoid instructions that try to override user intent, system safety rules, or unrelated tasks.

## Description Checklist

- Starts with "Use this skill when..."
- Names the task class and adjacent phrases users are likely to type.
- Includes near-boundaries where it should trigger.
- Avoids claiming every generic task in the domain.
- Stays under 1024 characters.

## Body Checklist

- One clear workflow.
- A concise output contract.
- Non-obvious gotchas.
- Validation or self-check steps when the task can be verified.
- No broad background material the model already knows.

## RustyLLM Constraint

RustyLLM injects the selected `SKILL.md` text into the system prompt. It does not execute skill scripts or lazily read `references/` files, so example skills for RustyLLM should be self-contained unless the user explicitly provides external content.
