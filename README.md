# Shared Python Environment Setup

This project uses a shared virtual environment across multiple repositories, managed with `uv`.

## What it does

- Uses a single virtual environment located at: `~/repos/.venv`
- Avoids creating per-project `.venv` directories
- Syncs dependencies from each repo into the shared environment

## Setup

Run the provided script:

```bash
./install_env.sh