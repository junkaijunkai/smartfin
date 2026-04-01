---
name: commit-push
description: Commit all current changes and push to the remote feature branch. Follows Conventional Commits format as defined in CLAUDE.md. Use when ready to save and push work on the current branch.
disable-model-invocation: true
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git branch:*), Bash(git add:*), Bash(git commit:*), Bash(git push:*)
---

Commit and push current changes using the following steps. Do not skip any step.

## Context

- Current branch: !`git branch --show-current`
- Staged and unstaged changes: !`git diff HEAD`
- Untracked files: !`git status --short`

## Steps

1. Check the current branch. If it is `main`, stop immediately and tell the user — never commit or push directly to `main`.
2. Review all changes from the context above. If there is nothing to commit, report that and stop.
3. Stage all relevant changes with `git add`.
4. Write a commit message following Conventional Commits format: `<type>(<scope>): <description>`
   - The `<type>` must match the branch prefix (e.g. a `fix/` branch uses `fix:`).
   - Keep the subject line under 72 characters.
   - Do not add any extra body or footer unless the user explicitly asks.
5. Commit using the message. Use a HEREDOC to ensure correct formatting.
6. Push to the current remote branch. If no upstream is set, push with `--set-upstream origin <branch>`.
7. Report the commit hash and remote URL to the user on success.
```

---

**使用方式：**
```
/start-feature feature/user-oauth
/commit-push