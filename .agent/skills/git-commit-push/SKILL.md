---
name: git-commit-push
description: Use when committing changes to the local repository and pushing to remote.
---

# Git Commit and Push

## Overview
This skill outlines the standard workflow for saving changes to the version control system. It emphasizes careful staging of files and descriptive commit messages.

## When to Use
- After completing a task or valid unit of work.
- When you want to save your progress to the remote repository.
- Before switching branches or pulling updates.

## Key Steps

1.  **Check Status**: Always start by checking which files have been modified or are untracked.
    ```bash
    git status
    ```

2.  **Stage Files**: Add specific files to the staging area. Avoid `git add .` unless you are certain you want to commit *all* changes (including untracked files).
    ```bash
    git add path/to/file1 path/to/file2
    ```

3.  **Commit**: Create a commit with a clear, descriptive message explaining *what* changed and *why*.
    ```bash
    git commit -m "feat: implement x feature"
    # or
    git commit -m "fix: resolve issue with y"
    ```

4.  **Push**: Push the commits to the current branch on the remote repository.
    ```bash
    git push origin <current-branch-name>
    ```
    *Tip: You can use `git push -u origin HEAD` to push to the current branch name automatically.*

## Common Pitfalls
-   **Accidental Adds**: Using `git add .` indiscriminately and committing temporary files, secrets, or build artifacts.
-   **Vague Messages**: "Fixed stuff" or "Update" are not helpful. Use [Conventional Commits](https://www.conventionalcommits.org/) format if possible (e.g., `feat:`, `fix:`, `docs:`).
-   **Detached HEAD**: Attempting to commit when not on a branch.
