Feature: Model initialization

  Scenario: Ollama Qwen model can initialize
    Given a model configuration file exists
    When I create the model client
    Then the model should be ready