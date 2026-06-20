---
name: rust-code-review
description: Use this skill when reviewing Rust code, diagnosing borrow checker errors, checking unsafe code, improving Cargo feature boundaries, or evaluating performance-sensitive Rust changes. Also use it when the user asks for a Rust review, Rust bug hunt, compiler error explanation, or refactoring guidance.
---

# Rust Code Review

## Workflow

1. Start with behavioral correctness: ownership, lifetimes, trait bounds, panic paths, integer bounds, UTF-8 boundaries, IO errors, and concurrency assumptions.
2. Check API compatibility next: public type changes, feature gates, target-specific code, error messages, and default behavior.
3. Review performance only after correctness: unnecessary allocation, repeated filesystem scans, lock scope, clone-heavy paths, avoidable parsing, and hot-loop formatting.
4. If code was changed, ask for or infer the verification surface: unit tests, feature-specific checks, target-specific checks, and smoke tests for CLI/server paths.

## Output

Lead with findings. For each finding, include:

- severity
- file/function or code area
- concrete failure mode
- minimal fix direction

If no major issue is visible, say so and mention remaining test gaps.

## Gotchas

- Do not recommend broad rewrites when a local fix preserves existing behavior.
- Treat `unsafe`, `std::fs`, network, subprocess, and lock-order changes as higher risk.
- For CLI tools, verify both one-shot and long-running modes when state is added.
- For feature-gated Rust, check at least the default feature set and the narrow feature set touched by the change.
