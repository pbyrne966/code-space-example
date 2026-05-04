#!/usr/bin/env bash
set -euo pipefail

docker compose build
uv run pytest tests/unit -v
uv run pytest tests/bdd_test -v
