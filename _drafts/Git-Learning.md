# Basic

## Local

### `git status`

- `Untracked files`: New add file and not be untracked

- `Changes to be committed`: After use `git add <file>`, the file is tracked and staged

- `Changes not staged for commit`: Tracked file is changed, but not staged. Use `git add` to stage.

### `git add <file>`

Track new file or modified file, and stage file.

### `git commit -m <message>`

Commit staged file into `Repository`.

## Remote

### `git fetch [remote-name]`

Only Get files from remote to your repository. It will not change anythings.

### `git pull`

Get files from remote and merge into current branch.