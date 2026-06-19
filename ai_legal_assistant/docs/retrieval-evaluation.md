# Dense retrieval evaluation

This baseline evaluates the original user query without query expansion:

1. Validate and trim the query.
2. Format it with the Qwen3 retrieval instruction.
3. Embed it with the same Qwen3 embedding model used for the corpus.
4. Validate that the result has 1024 dimensions.
5. Query the `law_chunks_qwen3_06b` Qdrant collection with cosine similarity.
6. Calculate macro-averaged Recall@K and MRR@K from the labeled testset.

The corpus Parquet shards were embedded with SentenceTransformers and the Hugging Face
model `Qwen/Qwen3-Embedding-0.6B`, using `max_seq_length=768` and L2-normalized vectors.
The query baseline uses the same library, model, maximum length, and normalization. An
Ollama adapter remains available for collections that were embedded through Ollama, but
it must not be used to evaluate this collection as vector equivalence is not guaranteed.

## Testset

Use one JSON object per line in `data/eval/retrieval_testset.jsonl`. A case must use
exactly one relevance level.

Chunk-level example:

```json
{"id":"capital-001","query":"Thời hạn góp đủ vốn điều lệ là bao lâu?","relevant_chunk_ids":["actual-chunk-id"]}
```

Article-level example:

```json
{"id":"tax-001","query":"Doanh nghiệp phải đăng ký mã số thuế khi nào?","relevant_article_ids":["actual-article-id"]}
```

Use chunk-level labels for precise passage retrieval. Use article-level labels when
multiple clauses from the same article are acceptable. Do not mix both levels in one
case.

## Run

Inspect one query:

```powershell
python scripts/query_qdrant.py "Thời hạn góp đủ vốn điều lệ là bao lâu?" `
  --embedding-provider huggingface `
  --model "Qwen/Qwen3-Embedding-0.6B" `
  --top-k 10
```

Evaluate the labeled testset:

```powershell
python scripts/evaluate_retrieval.py `
  --embedding-provider huggingface `
  --model "Qwen/Qwen3-Embedding-0.6B" `
  --cutoffs 1,3,5,10,20
```

Use `--no-query-instruction` only when the original query embeddings were produced
without the Qwen3 query instruction. Compare both settings on the same gold testset if
that detail was not recorded during corpus embedding.

To evaluate conditional query expansion with weighted RRF, add `--expand-query`. Keep
the command without this flag as the dense baseline and compare both on the same testset.
