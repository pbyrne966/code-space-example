# ConvFinQA Report

## Method
The project implements a retrieval-augmented question-answering pipeline for ConvFinQA. Each financial record is converted into structured retrieval chunks, embedded, stored in Postgres/pgvector, and retrieved only within the same source record as the user question. This keeps retrieval focused on the relevant financial report rather than searching across unrelated records.

The chunking layer creates four main evidence types: pre-text windows, post-text windows, table-column chunks, and table-metric chunks. Table-column chunks group all metrics for a period, which helps with questions asking for values in a specific year or quarter. Table-metric chunks transpose the table so that one metric can be compared across periods, which helps with trend or change questions. The pre-text and post-text are split into overlapping sentence windows so the retriever can still surface narrative evidence around the table.

At query time, the service extracts period hints from the question, retrieves matching vector chunks for the selected record, and builds a prompt containing the retrieved context plus a structured list of available numeric table-value candidates. The model is required to return JSON containing a scalar answer, supporting chunk citations, and, when arithmetic is needed, a calculation program that references those table-value IDs.

For direct lookup questions, the answer can be copied directly from the retrieved evidence. For arithmetic questions, the model selects the relevant values and expresses the calculation as structured operations rather than free-form reasoning text. The system parses the JSON response, validates it against the expected schema, executes the calculation program, and formats the final scalar result. This separates value selection from arithmetic execution, making the calculation trace easier to inspect and reducing reliance on the model to do arithmetic correctly in natural language.

## Evaluation

## Future Work

### Cache Invalidation
Caching is another area that could be strengthened. At present, cached answers are not fully tied to the conditions under which they were generated—such as the model used, prompt version, retrieval settings, embedding model, source record ID, or cited chunks. Without this context, a cached response may appear valid even after underlying configurations have changed, leading to stale or misleading results.

A more reliable approach would link each cached answer to a clear fingerprint of its retrieval context, along with any associated calculation trace. The system should only reuse cached outputs that still match these conditions and have passed validation. This would help ensure that cached responses remain accurate, consistent, and trustworthy over time.

### Prompt Versioning
The current prompt is embedded directly within the RAG service and tailored to the Qwen model. While this works for now, it limits flexibility and assumes that other models will behave in the same way. A more robust long-term approach would introduce a small, versioned prompt store, with prompts mapped to specific supported models. This would make it easier to tune prompts for each model, compare different prompt versions, and avoid relying on the assumption that all future models will handle JSON formatting, citations, and calculation-program instructions consistently.

### Citation Validation
Right now, the system asks the model to include chunk IDs as citations, but it doesn’t verify them afterward. This creates a risk that citations may be incorrect or unsupported. At a minimum, a validation step should confirm that every cited chunk ID was actually retrieved during the search process. A more robust approach would go further by checking that each cited chunk truly contains the specific evidence used in the answer—such as the referenced table value, metric, year, or supporting text. If citations can’t be validated, the system could either reject the response or trigger a retry to ensure accuracy.

### Model Compatibility
The current implementation has been tested primarily with the Qwen/Ollama setup. While this is sufficient for the present scope, it does not yet demonstrate that the system is truly model-agnostic. Before expanding support to additional model backends, future work should introduce a lightweight compatibility suite. This suite would verify key behaviours across models, including JSON validity, consistent scalar answer formatting, reliable citation handling, correct generation of calculation programs, and stable, deterministic outputs.

### Deployment Architecture
The current implementation already separates chunking, retrieval, model access, answer generation, and calculation execution into distinct service layers, with chunked evidence and embeddings stored through Postgres/pgvector. A future production deployment could extend this into a more explicitly service-oriented RAG system. The ingestion pipeline would run separately from the online answering path, while a lightweight API service would handle user questions by retrieving relevant chunks, selecting the appropriate prompt version for the configured model, generating a structured answer, validating citations, executing any calculation program, and caching only validated results. This separation would make the system easier to reproduce, monitor, and extend to additional models or datasets.

```text
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
```

A natural deployment path would be to containerise the API layer and run multiple stateless API replicas behind a load balancer. These API containers would connect to a shared datastore containing source records, retrieval chunks, embeddings, chat history, cached answers, citation metadata, and prompt/version information. This design would allow the answering layer to scale independently from the storage layer. It would also make it possible to support multiple model backends and embedding architectures side by side: for example, one deployment could route requests to a local Qwen/Ollama backend, while another could use a hosted model or a different embedding model. The prompt store would provide the matching prompt template for each supported model and task, helping avoid hard-coding one prompt for all models. Because embeddings are stored with model provenance, the retriever could select the correct vector index for the active embedding architecture.


For a more scalable deployment, the system could use the following structure:

```text
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
```



## [may not apply] If & how you've used coding assistants or gen AI tools to help with this assignment
I used AI assistance for generating tests, reviewing code, and getting feedback on the report wording. The core implementation decisions and service code were written by me, especially in the model service, chunking service, data service, and aggregator service.
