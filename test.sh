#!/usr/bin/env bash
set -euo pipefail

docker compose build
docker compose run --rm app uv run pytest tests/unit -v
docker compose run --rm app uv run pytest tests/bdd_test -v
