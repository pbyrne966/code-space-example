Feature: Full stack RAG answering

  Scenario: Application answers for a random ingested record
    Given the configured full application stack is loaded
    And the configured ConvFinQA data has been ingested
    When I ask a question for a random ingested record
    Then the application should return a RAG answer
