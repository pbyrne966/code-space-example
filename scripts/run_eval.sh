#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_CONFIG_PATH:=configs/model_config.toml}"
: "${RAW_DATA_PATH:=data/evaluation/convfinqa_train_subset.json}"
: "${INGESTION_FILE_PATHS:=data/evaluation/convfinqa_train_subset.json,data/evaluation/convfinqa_context_window_mock.json}"
: "${POSTGRES_HOST:=localhost}"
: "${POSTGRES_PORT:=5432}"
: "${POSTGRES_DB:=convfinqa}"
: "${POSTGRES_USER:=convfinqa}"
: "${POSTGRES_PASSWORD:=convfinqa}"
: "${CACHING_ON:=False}"

: "${LIMIT:=10}"
: "${NUMERIC_TOLERANCE:=0.001}"
: "${OUTPUT:=data/evaluation/evaluation_results.json}"
: "${CONTEXT_WINDOW_DATA_PATH:=data/evaluation/convfinqa_context_window_mock.json}"
: "${UNRELATED_QUESTIONS_PATH:=data/evaluation/raw_quistions.json}"

export MODEL_CONFIG_PATH
export RAW_DATA_PATH
export INGESTION_FILE_PATHS
export POSTGRES_HOST
export POSTGRES_PORT
export POSTGRES_DB
export POSTGRES_USER
export POSTGRES_PASSWORD
export CACHING_ON

uv run python scripts/evaluate_rag.py \
  --split train \
  --limit "$LIMIT" \
  --numeric-tolerance "$NUMERIC_TOLERANCE" \
  --output "$OUTPUT" \
  --context-window-data-path "$CONTEXT_WINDOW_DATA_PATH" \
  --unrelated-questions-path "$UNRELATED_QUESTIONS_PATH"
