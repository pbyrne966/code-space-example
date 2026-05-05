from pathlib import Path
import os

import pytest
from pytest_bdd import given, scenario, then, when

from src.model_service.models import BaseModelFactory
from tests.unit_tests.mock_settings import MockSettings


@scenario("model_initialization.feature", "Ollama Qwen model can initialize")
def test_model_initialization():
    """Run the model initialization scenario."""
    pass


@given("a model configuration file exists", target_fixture="model_config_path")
def model_config_path():
    """Provide a stable model config path for the scenario."""
    settings = MockSettings()
    path = Path(settings.model_config_path)

    assert path.exists()
    return path


@when("I create the model client", target_fixture="model_client")
def model_client(model_config_path):
    """Create the Ollama model client using the sample config."""
    if os.getenv("RUN_MODEL_INTEGRATION") != "1":
        pytest.skip("Set RUN_MODEL_INTEGRATION=1 to run live model initialization")

    return BaseModelFactory.create(
        model_type="QWEN",
        model_config_path=model_config_path,
    )


@then("the model should be ready")
def model_should_be_ready(model_client):
    """Assert the model client reached the ready state."""
    assert model_client.is_model_ready is True
