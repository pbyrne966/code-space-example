# ConvFinQA Report

*The headers here are guidelines, you can structure your report however you like.*
## Method
The project builds a question answering pipeline for ConvFinQA. The main idea is to turn
each financial record into evidence chunks, retrieve the most relevant chunks for a user
question, and then ask the model to answer using only that retrieved evidence.

First, each ConvFinQA record is processed into retrieval chunks. The table is represented
in two ways: by table column and by table metric. This helps the system answer both
period-based questions, where the user asks about a value in a specific year or period,
and metric-based questions, where the user asks about how one financial metric changes
across periods. The pre-text and post-text are also chunked into sentence windows so that
the system can use surrounding narrative context from the source document.

The benchmark dialogue is deliberately not chunked. This means the system does not index
the original ConvFinQA questions, gold answers, or gold calculation programs. This is
important because including them would risk leaking the answer into the retrieval context
and would make the task much less meaningful.

When a user asks a question, the system retrieves evidence chunks from the same source
record. These chunks are inserted into a prompt along with structured table values that
the model is allowed to use. The model is instructed to return a single structured answer:
a scalar answer, a list of supporting citations, and, when arithmetic is needed, a
calculation program.

For direct lookup questions, the model can copy the answer from the retrieved evidence.
For arithmetic questions, the model does not just return a free-text calculation. Instead,
it describes the calculation as a sequence of operations over retrieved table values. The
system then executes that calculation itself and formats the final answer. This separates
reasoning about which values to use from the arithmetic execution, making the final result
easier to inspect and less dependent on the model doing maths correctly in prose.

The project also includes a chat workflow so that a user can ask questions against a
selected ConvFinQA record, see the answer, view the calculation trace where available, and
inspect the cited chunks. Previously seen questions can be cached, although the current
cache is simple and is discussed further in the future work section.
## Error Analysis
Lorem ipsum dolor sit amet consectetur adipiscing elit
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
