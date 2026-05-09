Feature: Postgres sample ingestion

  Scenario: ProcessLayer ingests a sample file into Postgres
    Given a Postgres behaviour database is configured
    And a raw ConvFinQA file built from one sample record
    When I run the process layer ingestion
    Then chunks should be persisted for the sample record
    And vector retrieval should return persisted chunks

  Scenario: Chat session records messages after sample ingestion
    Given a Postgres behaviour database is configured
    And a raw ConvFinQA file built from one sample record
    And the sample record has been ingested
    When I start a chat session for the sample record
    And I append a user question and assistant answer
    Then the chat session should track both messages
    And chat history should contain the user answer pair

  Scenario: Cached answer is replayed as a fresh chat exchange
    Given a Postgres behaviour database is configured
    And a raw ConvFinQA file built from one sample record
    And the sample record has been ingested
    When I start a chat session for the sample record
    And I ask the same cacheable question twice
    Then the second answer should come from the answer cache
    And chat history should contain two cache replay pairs
    And answer cache should remain separate from chat history

  Scenario: Period-filtered retrieval handles date combinations
    Given a Postgres behaviour database is configured
    And a raw ConvFinQA file built from period-rich records
    When I run the period-rich process layer ingestion
    Then retrieval by year and month should return matching chunks
    And retrieval by quarter and month should return matching chunks
    And retrieval by exact date should return matching chunks
