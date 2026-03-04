# Hybrid Query Patterns

This document provides query examples and patterns for Chapter 3: Hybrid Search with Learned Reranking.

## Table of Contents

- [Hybrid Queries](#hybrid-queries)
- [Query Components](#query-components)
- [Query Patterns](#query-patterns)
- [Testing Queries](#testing-queries)
- [Debugging Queries](#debugging-queries)
- [Performance Tuning](#performance-tuning)

---

## Hybrid Queries

### Basic Hybrid Query

Combines lexical and semantic search using OR operator:

**HTTP Request:**
```http
POST https://<your-endpoint>/search/
Content-Type: application/json

{
  "yql": "select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR ({targetHits:100}nearestNeighbor(Description_embedding,q_embedding)) OR userQuery()",
  "query": "comfortable running shoes",
  "input.query(q_embedding)": "embed(@query)",
  "ranking.profile": "hybrid",
  "hits": 20
}
```

**Vespa CLI:**
```bash
vespa query \
  'yql=select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR ({targetHits:100}nearestNeighbor(Description_embedding,q_embedding)) OR userQuery()' \
  'query=comfortable running shoes' \
  'input.query(q_embedding)=embed(@query)' \
  'ranking.profile=hybrid' \
  'hits=20'
```

### With Reranking

Use the `rerank` profile for ML-based second-phase ranking:

**HTTP Request:**
```http
POST https://<your-endpoint>/search/
Content-Type: application/json

{
  "yql": "select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()",
  "query": "blue jeans for women",
  "input.query(q_embedding)": "embed(@query)",
  "ranking.profile": "rerank",
  "hits": 20
}
```

**Vespa CLI:**
```bash
vespa query \
  'yql=select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()' \
  'query=blue jeans for women' \
  'input.query(q_embedding)=embed(@query)' \
  'ranking.profile=rerank' \
  'hits=20'
```

---

## Query Components

### YQL Query Operators

**1. nearestNeighbor (Semantic Search)**

```yql
{targetHits:100}nearestNeighbor(ProductName_embedding, q_embedding)
```

- **`targetHits`**: Number of approximate nearest neighbors to retrieve
- **`ProductName_embedding`**: Document embedding field
- **`q_embedding`**: Query embedding (passed via `input.query(...)`)

**2. userQuery (Lexical Search)**

```yql
userQuery()
```

- Searches the default fieldset (ProductName, Description)
- Uses query text from `query` parameter
- Applies BM25/nativeRank scoring

**3. OR Operator (Combine Results)**

```yql
nearestNeighbor(...) OR userQuery()
```

- Combines result sets (union)
- Each document scored by rank profile
- Duplicates removed automatically

### Query Parameters

**Required parameters:**
```json
{
  "yql": "...",                                  // YQL query
  "query": "search text",                        // Query text for lexical search
  "input.query(q_embedding)": "embed(@query)",   // Embed query text
  "ranking.profile": "hybrid"                    // Rank profile to use
}
```

**Optional parameters:**
```json
{
  "hits": 20,                                    // Number of results to return
  "offset": 0,                                   // Pagination offset
  "timeout": 5000,                               // Query timeout (ms)
  "summary": "default",                          // Summary format
  "presentation.format": "json"                  // Response format
}
```

### Embedding Query Text

**Automatic embedding:**
```json
{
  "query": "comfortable running shoes",
  "input.query(q_embedding)": "embed(@query)"
}
```

The `embed(@query)` function:
- Takes query text from `query` parameter
- Uses embedder configured in services.xml (`arctic`)
- Returns 384-dimensional vector
- Passed to rank profile as `q_embedding`

---

## Query Patterns

### Pattern 1: Hybrid Search (Balanced)

**Use case**: General search, balanced between precision and recall

```json
{
  "yql": "select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR ({targetHits:100}nearestNeighbor(Description_embedding,q_embedding)) OR userQuery()",
  "query": "home decor items",
  "input.query(q_embedding)": "embed(@query)",
  "ranking.profile": "hybrid",
  "hits": 20
}
```

**Characteristics:**
- Searches both embeddings (name + description)
- Combines with lexical search
- Good for diverse queries

### Pattern 2: Semantic-Heavy Hybrid

**Use case**: Natural language queries, emphasize semantic matching

```json
{
  "yql": "select * from product where ({targetHits:200}nearestNeighbor(ProductName_embedding,q_embedding)) OR ({targetHits:200}nearestNeighbor(Description_embedding,q_embedding)) OR ({defaultIndex:'ProductName'}userQuery())",
  "query": "gift ideas for fitness enthusiasts",
  "input.query(q_embedding)": "embed(@query)",
  "ranking.profile": "hybrid",
  "hits": 20
}
```

**Characteristics:**
- Higher `targetHits` for semantic search (200)
- Lexical search limited to ProductName
- Better for conceptual/natural language queries

### Pattern 3: Lexical-Heavy Hybrid

**Use case**: Exact matches important (brand names, product codes)

```json
{
  "yql": "select * from product where ({targetHits:50}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()",
  "query": "Nike Air Max 270",
  "input.query(q_embedding)": "embed(@query)",
  "ranking.profile": "hybrid",
  "hits": 20
}
```

**Characteristics:**
- Lower `targetHits` for semantic search (50)
- Full lexical search on default fieldset
- Better for exact-match queries

### Pattern 4: Filtered Hybrid Search

**Use case**: Search within specific category or attribute

```json
{
  "yql": "select * from product where (({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()) AND Gender='Women' AND Price < 5000",
  "query": "summer dress",
  "input.query(q_embedding)": "embed(@query)",
  "ranking.profile": "hybrid",
  "hits": 20
}
```

**Characteristics:**
- Hybrid search with filters
- Filters applied after retrieval (AND operator)
- Efficient for category-specific searches

### Pattern 5: Reranked Hybrid Search

**Use case**: Production search with ML reranking

```json
{
  "yql": "select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()",
  "query": "winter jacket",
  "input.query(q_embedding)": "embed(@query)",
  "ranking.profile": "rerank",
  "hits": 20
}
```

**Characteristics:**
- First-phase: Fast hybrid ranking
- Second-phase: LightGBM reranking on top 20
- Best quality, slightly slower

---

## Testing Queries

### Using HTTP REST Client (VS Code Extension)

Create a `queries.http` file:

```http
### 1. Hybrid search - basic
POST https://{{endpoint}}/search/
Content-Type: application/json

{
  "yql": "select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()",
  "query": "blue t-shirt",
  "input.query(q_embedding)": "embed(@query)",
  "ranking.profile": "hybrid",
  "hits": 10
}

### 2. Hybrid search with reranking
POST https://{{endpoint}}/search/
Content-Type: application/json

{
  "yql": "select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()",
  "query": "comfortable shoes",
  "input.query(q_embedding)": "embed(@query)",
  "ranking.profile": "rerank",
  "hits": 10
}

### 3. Lexical only (baseline comparison)
POST https://{{endpoint}}/search/
Content-Type: application/json

{
  "yql": "select * from product where userQuery()",
  "query": "red dress",
  "ranking.profile": "default",
  "hits": 10
}

### 4. Semantic only (baseline comparison)
POST https://{{endpoint}}/search/
Content-Type: application/json

{
  "yql": "select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding))",
  "approximate_query_string": "elegant evening gown",
  "input.query(q_embedding)": "embed(@approximate_query_string)",
  "ranking.profile": "closeness_productname_description",
  "hits": 10
}
```

### Using Vespa CLI

**1. Basic hybrid search:**
```bash
vespa query \
  'yql=select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()' \
  'query=blue t-shirt' \
  'input.query(q_embedding)=embed(@query)' \
  'ranking.profile=hybrid'
```

**2. With filters:**
```bash
vespa query \
  'yql=select * from product where (({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()) AND Gender="Women"' \
  'query=summer dress' \
  'input.query(q_embedding)=embed(@query)' \
  'ranking.profile=hybrid'
```

**3. Compare ranking profiles:**
```bash
# Hybrid baseline
vespa query 'yql=...' 'ranking.profile=hybrid' > hybrid_results.json

# With reranking
vespa query 'yql=...' 'ranking.profile=rerank' > rerank_results.json

# Compare
diff hybrid_results.json rerank_results.json
```

### Using Python/pyvespa

```python
from vespa.application import Vespa

# Connect to Vespa
app = Vespa(url="https://your-endpoint", cert="/path/to/cert.pem", key="/path/to/key.pem")

# Hybrid search
response = app.query(
    yql="select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()",
    query="comfortable shoes",
    ranking="hybrid",
    body={
        "input.query(q_embedding)": "embed(@query)"
    }
)

# Print results
for hit in response.hits:
    print(f"{hit['fields']['ProductName']}: {hit['relevance']}")
```

---

## Debugging Queries

### Expose Summary Features

Add `summary-features` to see intermediate ranking scores:

```json
{
  "yql": "select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()",
  "query": "running shoes",
  "input.query(q_embedding)": "embed(@query)",
  "ranking.profile": "hybrid",
  "hits": 5,
  "presentation.format.tensors": "short"
}
```

**Response includes:**
```json
{
  "root": {
    "children": [
      {
        "id": "...",
        "relevance": 4.567,
        "fields": {
          "ProductName": "Nike Air Zoom Pegasus",
          "summaryfeatures": {
            "native_rank_name": 2.34,
            "native_rank_description": 1.89,
            "closeness_productname": 0.87,
            "closeness_description": 0.72
          }
        }
      }
    ]
  }
}
```

**Analyzing summary features:**
- High `native_rank_name`: Strong lexical match on product name
- High `closeness_productname`: Strong semantic match on embedding
- Use this to understand why documents rank high/low

### Trace Query Execution

Enable query tracing to see detailed execution:

```json
{
  "yql": "select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()",
  "query": "shoes",
  "input.query(q_embedding)": "embed(@query)",
  "ranking.profile": "hybrid",
  "hits": 5,
  "trace.level": 5
}
```

**Trace information includes:**
- Query parsing
- Retrieval statistics
- Ranking phase timings
- Feature computation

### Compare Ranking Profiles

**Script to compare profiles:**
```python
import json

query_text = "comfortable running shoes"

profiles = ["default", "closeness_productname_description", "hybrid", "rerank"]

results = {}
for profile in profiles:
    response = app.query(
        yql="select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()",
        query=query_text,
        ranking=profile,
        body={"input.query(q_embedding)": "embed(@query)"},
        hits=10
    )
    results[profile] = [(hit['fields']['ProductName'], hit['relevance'])
                        for hit in response.hits]

# Compare top 5 results
for profile in profiles:
    print(f"\n{profile}:")
    for name, score in results[profile][:5]:
        print(f"  {score:.3f}: {name}")
```

---

## Performance Tuning

### targetHits Tuning

**Impact of targetHits:**
- Higher values: Better recall, slower queries
- Lower values: Faster queries, may miss results

**Recommendations:**
```json
// Fast, may miss some results
{targetHits: 50}

// Balanced (recommended)
{targetHits: 100}

// High recall, slower
{targetHits: 200}

// Maximum recall, slowest
{targetHits: 500}
```

**Measuring impact:**
```bash
# Test with different targetHits
for hits in 50 100 200 500; do
  echo "targetHits: $hits"
  time vespa query \
    "yql=select * from product where ({targetHits:${hits}}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()" \
    'query=shoes' \
    'input.query(q_embedding)=embed(@query)' \
    'ranking.profile=hybrid'
done
```

### Query Timeout

Set appropriate timeout for complex queries:

```json
{
  "yql": "...",
  "timeout": 5000  // 5 seconds
}
```

**Recommendations:**
- Simple queries: 1000-2000ms
- Hybrid queries: 2000-5000ms
- With reranking: 3000-10000ms

### Caching

Vespa caches:
- Query results (cache-control headers)
- Embedding computations (automatic)
- Model predictions (automatic)

**Enable result caching:**
```bash
vespa query '...' --header="Cache-Control: max-age=60"
```

### Parallel Queries

For multiple queries, use parallel execution:

```python
from concurrent.futures import ThreadPoolExecutor

queries = ["shoes", "dress", "jacket", "pants", "shirt"]

def run_query(query_text):
    return app.query(
        yql="select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()",
        query=query_text,
        ranking="hybrid",
        body={"input.query(q_embedding)": "embed(@query)"}
    )

with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(run_query, queries))
```

---

## Summary

**Key patterns:**
- **Basic hybrid**: OR combination of semantic + lexical
- **Filtered hybrid**: Add AND filters for category search
- **Reranked hybrid**: Use ML model for second-phase ranking

**Debugging tools:**
- `summary-features`: See intermediate scores
- `trace.level`: Detailed execution trace
- Profile comparison: Compare different ranking approaches

**Performance tips:**
- Tune `targetHits` based on latency/quality trade-off
- Set appropriate timeouts
- Use caching for repeated queries
- Parallelize multiple queries

**Next steps:**
- Test queries with your data
- Measure latency and quality
- Compare ranking profiles
- Train and deploy reranker
- Iterate on features and model

**Resources:**
- [Vespa Query API](https://docs.vespa.ai/en/query-api.html)
- [YQL Reference](https://docs.vespa.ai/en/query-language.html)
- [Ranking Documentation](https://docs.vespa.ai/en/ranking.html)
