from src.model_service.models import ModelConfig, OllamaQwenClient


def test_extract_output_uses_ollama_top_level_token_counts() -> None:
    client = OllamaQwenClient(
        ModelConfig(
            base_url="http://example.com",
            model_name="qwen",
            chat_endpoint="/api/chat",
            max_tokens=32,
            allowed_timeout=5,
        )
    )

    output = client._extract_output(
        {
            "message": {
                "role": "assistant",
                "content": "ready",
            },
            "prompt_eval_count": 11,
            "eval_count": 3,
        }
    )

    assert output.output == "ready"
    assert output.tokens_used == 14


def test_extract_output_defaults_missing_ollama_token_counts_to_zero() -> None:
    client = OllamaQwenClient(
        ModelConfig(
            base_url="http://example.com",
            model_name="qwen",
            chat_endpoint="/api/chat",
            max_tokens=32,
            allowed_timeout=5,
        )
    )

    output = client._extract_output(
        {
            "message": {
                "role": "assistant",
                "content": "ready",
            },
        }
    )

    assert output.tokens_used == 0
