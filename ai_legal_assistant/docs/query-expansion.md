# Legal query expansion

The query planning pipeline uses the local Hugging Face generation model
`Qwen/Qwen3-0.6B`. It is separate from `Qwen/Qwen3-Embedding-0.6B`, which remains
responsible only for query vectors.

```text
raw query
  -> Vietnamese normalization
  -> Qwen3 query analysis and conditional expansion
  -> evidence and drift validation
  -> QueryPlan
  -> per-query dense HNSW and BM25 retrieval (oversampled)
  -> runtime content identity and per-list deduplication
  -> per-scope and global weighted reciprocal-rank fusion
  -> cross-encoder intent-aware reranking
  -> scope-aware selection and per-article cap
```

## QueryPlan rules

- The normalized original query is always first with weight `1.0`.
- At most three semantic or scope variants are retained.
- At most two subqueries are retained for a multi-issue question.
- Exact article or clause lookups do not receive semantic expansion.
- Explicit company types cannot expand into another company type.
- Negation must be preserved.
- Expansion cannot invent article numbers, deadlines, penalty amounts, or other numeric facts.
- HyDE is not used.
- Extracted entities and explicit constraints must include an evidence quote that occurs
  in the normalized query.
- Fabricated evidence or inconsistent analysis fails closed.
- If validated analysis succeeds but expansion JSON or variants fail safety checks, retrieval
  continues with the original query only and records a warning in `QueryPlan`.

Qwen performs query analysis and expansion in two focused structured-JSON calls. The
first call only extracts analysis/evidence; exact lookups stop there. The second call only
generates variants or subqueries from validated analysis. No rule-based analyzer or
rule-based expander participates in runtime planning.

## Inspect a plan

```powershell
python scripts/plan_legal_query.py `
  "Thời hạn góp đủ vốn điều lệ là bao lâu?" `
  --planner-model "Qwen/Qwen3-0.6B"
```

## Retrieve with automatic expansion

Interactive retrieval uses conditional query planning by default. The planner analyzes the
query and exact lookups remain single-query searches; other query types are expanded only
when the validated plan calls for it:

```powershell
python scripts/query_qdrant.py `
  "Thời hạn góp đủ vốn điều lệ là bao lâu?" `
  --per-query-top-k 20 `
  --top-k 5
```

Use `--retrieval-mode baseline` only when a single-query baseline is explicitly needed.

Each semantic query is embedded with the same `Qwen/Qwen3-Embedding-0.6B` pipeline as
the corpus and is also searched against the BM25 index. Validated lexical terms are added
to the sparse query when absent from the semantic query. Dense and sparse lists are
deduplicated by `chunk_id` and fused with weighted RRF. The original query has weight
`1.0`; BM25 contributes with weight `0.7` by default.

Use `--disable-bm25` for a dense-only interactive comparison. Build a missing index with
`python scripts/build_bm25_index.py`.

Runtime retrieval requests `per_query_top_k * oversample_factor` hits from each modality,
then keeps unique content using a stable hash of `article_id` and normalized text. This
avoids relying on legacy `chunk_id` values that collide in the current corpus. Ambiguous
plans reserve candidates from every validated scope before reranking. Final selection
keeps at most one chunk per article for ambiguous queries (two for other query types).
For ambiguous queries with an enabled reranker, additional hits must remain within the
configured `scope_relevance_margin` of the weakest required-scope result. The system may
therefore return fewer than `top_k` results rather than fill the context with weak evidence.

The interactive command enables `BAAI/bge-reranker-v2-m3` by default. Use
`--disable-reranker` only for A/B evaluation or when the reranker model is unavailable.
The final hit metadata includes `content_id`, `legacy_chunk_ids`, `matched_scopes`,
`rrf_score`, and `rerank_score`.

## A/B evaluation

Baseline:

```powershell
python scripts/evaluate_retrieval.py --cutoffs 1,3,5,10,20
```

Expanded:

```powershell
python scripts/evaluate_retrieval.py `
  --cutoffs 1,3,5,10,20 `
  --expand-query `
  --per-query-top-k 20
```

Compare both runs on the same labeled testset. Do not judge expansion quality from a
single query; use a stratified testset containing exact lookups, legal concepts, legal
situations, multi-issue questions, ambiguous questions, and negative statements.

The local machine currently has no CUDA device. Loading and generating with the planner
on CPU can take several minutes per query. Production should keep both models loaded in
a long-running process and use a CUDA device or a dedicated model server.

`Qwen/Qwen3-0.6B` has been observed to ignore multi-branch expansion constraints and to
invent duration values for an ambiguous capital-contribution query. These outputs are
rejected; retrieval safely uses only the original query and never falls back to rule-based
expansion. A larger instruction model is required if this behavior persists on the evaluation set.
