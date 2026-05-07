# ConvFinQA Report

*The headers here are guidelines, you can structure your report however you like.*
## Method
The project implements a question-answering pipeline for ConvFinQA. The core idea is to transform each financial record into structured evidence chunks, retrieve the most relevant pieces for a given query, and generate answers strictly grounded in that retrieved evidence.

Each ConvFinQA record is first processed into retrieval chunks. The table is represented in two complementary ways: by column and by metric. This enables the system to handle both period-based questions (e.g., values for a specific year) and metric-based questions (e.g., how a financial metric changes over time). In addition, the pre-text and post-text are segmented into sentence-level windows, allowing the model to incorporate relevant narrative context from the source document.

At query time, relevant chunks are retrieved from the same source record and inserted into a prompt, along with structured table values available for use. The model is required to produce a single structured output consisting of a scalar answer, supporting citations, and—when necessary—a calculation program.

For direct lookup questions, the answer can be copied directly from the retrieved evidence. For arithmetic questions, the model does not generate free-form calculations. Instead, it represents the solution as a sequence of operations over retrieved values. This program is then executed by the system, which produces the final result. By separating value selection from arithmetic execution, this approach improves transparency and reduces reliance on the model’s ability to perform calculations correctly in natural language.

## Evaluation

## Future Work

Caching.
Caching is another area that could be strengthened. At present, cached answers are not fully tied to the conditions under which they were generated—such as the model used, prompt version, retrieval settings, embedding model, source record ID, or cited chunks. Without this context, a cached response may appear valid even after underlying configurations have changed, leading to stale or misleading results.

A more reliable approach would link each cached answer to a clear fingerprint of its retrieval context, along with any associated calculation trace. The system should only reuse cached outputs that still match these conditions and have passed validation. This would help ensure that cached responses remain accurate, consistent, and trustworthy over time.

Prompt design.
The current prompt is embedded directly within the RAG service and tailored to the Qwen model. While this works for now, it limits flexibility and assumes that other models will behave in the same way. A more robust long-term approach would introduce a small, versioned prompt store, with prompts mapped to specific supported models. This would make it easier to tune prompts for each model, compare different prompt versions, and avoid relying on the assumption that all future models will handle JSON formatting, citations, and calculation-program instructions consistently.

Citation validation.
Right now, the system asks the model to include chunk IDs as citations, but it doesn’t verify them afterward. This creates a risk that citations may be incorrect or unsupported. At a minimum, a validation step should confirm that every cited chunk ID was actually retrieved during the search process. A more robust approach would go further by checking that each cited chunk truly contains the specific evidence used in the answer—such as the referenced table value, metric, year, or supporting text. If citations can’t be validated, the system could either reject the response or trigger a retry to ensure accuracy.

The current implementation has been tested primarily with the Qwen/Ollama setup. While this is sufficient for the present scope, it does not yet demonstrate that the system is truly model-agnostic. Before expanding support to additional model backends, future work should introduce a lightweight compatibility suite. This suite would verify key behaviours across models, including JSON validity, consistent scalar answer formatting, reliable citation handling, correct generation of calculation programs, and stable, deterministic outputs.

Archetetcure
After addressing prompt versioning, validated caching, citation checking, and model compatibility, this project could be deployed as a service-oriented RAG system. The ingestion pipeline would run separately from the online answering path, storing chunked evidence and embeddings in Postgres/pgvector. A lightweight API service would handle user questions by retrieving relevant chunks, selecting the appropriate prompt version for the configured model, generating a structured answer, validating citations, executing any calculation program, and caching only validated results. This separation would make the system easier to reproduce, monitor, and extend to additional models or datasets.

Client / CLI / API caller
        |
        v
RAG API service
        |
        |-- Prompt store
        |-- Cache service
        |-- Citation validator
        |-- Calculation executor
        |
        v
Retriever / Postgres + pgvector
        |
        v
Model backend, e.g. Ollama/Qwen or hosted LLM

A natural deployment path would be to containerise the API layer and run multiple stateless API replicas behind a load balancer. These API containers would connect to a shared datastore containing source records, retrieval chunks, embeddings, chat history, cached answers, citation metadata, and prompt/version information. This design would allow the answering layer to scale independently from the storage layer. It would also make it possible to support multiple model backends and embedding architectures side by side: for example, one deployment could route requests to a local Qwen/Ollama backend, while another could use a hosted model or a different embedding model. The prompt store would provide the matching prompt template for each supported model and task, helping avoid hard-coding one prompt for all models. Because embeddings are stored with model provenance, the retriever could select the correct vector index for the active embedding architecture.


for a more scable deployment we could have something like this

              Load balancer
                   |
        -------------------------
        |           |           |
   API container API container API container
        |           |           |
        -------- Shared services --------
                   |
      --------------------------------
      |              |               |
Postgres/pgvector  Prompt store   Model backends
      |                              |
Embeddings/chunks              Qwen / other LLMs



## [may not apply] If & how you've used coding assistants or gen AI tools to help with this assignment
I have mainly used to generated tests and for code reviews.
I have written most of the code myself -> esspecially in the various services -> like model_service, chunking_service, data_service and aggregator_service
