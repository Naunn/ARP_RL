#!/usr/bin/env bash

SHARED_VENV="$HOME/repos/.venv"
REPO_PATH="$(pwd)"

# Step 1: Ensure shared venv exists
if [ ! -f "$SHARED_VENV/bin/activate" ]; then
    echo "Shared virtual environment not found at $SHARED_VENV. Creating..."
    python3 -m venv "$SHARED_VENV"
fi

# Step 2: Tell uv that the project environment is the shared venv
# This marks the shared venv as the project environment without creating a local .venv
export UV_PROJECT_ENVIRONMENT="$SHARED_VENV"

# If your uv version supports it, skip creation
# If not, activating and sync after setting UV_PROJECT_ENVIRONMENT is enough
# Do NOT run `uv venv $SHARED_VENV` here, as that triggers creation

# Step 3: Activate the shared venv
source "$SHARED_VENV/bin/activate"

# Step 4: Sync dependencies from this repo's .toml into the shared venv
uv sync --active --reinstall

echo "Shared environment activated and dependencies synced for $REPO_PATH"