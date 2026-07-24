#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "Fehler: uv wurde nicht gefunden. Installation: https://docs.astral.sh/uv/" >&2
    exit 1
fi

uv run python annotation/scratch_annotator.py
