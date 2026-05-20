# Project Setup

This project uses `pyproject.toml` and is managed with uv.

## Install uv

curl -LsSf https://astral.sh/uv/install.sh | sh

Verify:
uv --version

## Setup Project

From the repository root:

uv sync

This will create a local .venv and install all dependencies from pyproject.toml (using uv.lock if available).

## Activate Environment

macOS / Linux:
source .venv/bin/activate

Windows (PowerShell):
.venv\Scripts\Activate.ps1

## Run Project

uv run python main.py

## Add Dependencies

uv add <package>

## Update Environment

uv sync

## Notes

- Do not use pip install or python -m venv
- Always use uv sync to keep environment aligned with the project
- .venv/ should be gitignored