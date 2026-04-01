---
name: start-feature
description: Start a new feature branch. Syncs the latest code from remote, then creates and switches to a new branch following the project naming convention. Use when beginning any new piece of work.
disable-model-invocation: true
argument-hint: <type>/<description>  e.g. feature/user-oauth
allowed-tools: Bash(git pull:*), Bash(git checkout:*), Bash(git branch:*)
---

Start a new feature branch using the following steps. Do not skip any step.

## Context

- Current branch: !`git branch --show-current`
- Local status: !`git status --short`

## Steps

1. If the current branch is not `main`, warn the user and ask them to confirm before continuing.
2. Run `git pull --rebase` to sync the latest code from remote. If there are conflicts, stop and report them — do not proceed.
3. Create and switch to a new branch using the name provided in $ARGUMENTS.
   - The branch name must follow the convention `<type>/<description>` defined in CLAUDE.md.
   - Valid types: `feature/`, `fix/`, `hotfix/`, `chore/`, `docs/`.
   - If $ARGUMENTS is empty, ask the user for the branch name before proceeding.
4. Confirm the active branch after switching and report success to the user.