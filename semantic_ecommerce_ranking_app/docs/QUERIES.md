# Vector Search Query Guide

This document provides query examples and patterns for semantic/vector search in the semantic e-commerce ranking tutorial.

**References:**
- [Vespa Query API](https://docs.vespa.ai/en/query-api.html)
- [nearestNeighbor Reference](https://docs.vespa.ai/en/reference/query-language-reference.html#nearestneighbor)
- [YQL Reference](https://docs.vespa.ai/en/reference/query-language-reference.html)

---

## Table of Contents

1. [Vector Search Queries](#vector-search-queries)
2. [Embedding Query Text](#embedding-query-text)
3. [Multiple Embedding Fields](#multiple-embedding-fields)
4. [Combining Lexical and Vector Search](#combining-lexical-and-vector-search)
5. [Query Parameters](#query-parameters)
6. [Evaluation Framework](#evaluation-framework)
7. [Common Patterns](#common-patterns)
8. [Troubleshooting](#troubleshooting)

---

## Vector Search Queries

### Basic nearestNeighbor Query

**Single embedding field:**
```http
POST https://<mTLS_ENDPOINT_DNS_GOES_HERE>/search/
Content-Type: application/json

{
    "yql": "select * from product where {targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)",
    "ranking.profile": "closeness_productname_description",
    "approximate_query_string": "blue t-shirt",
    "input.query(q_embedding)": "embed(@approximate_query_string)"
}
```

**Via CLI:**
```bash
vespa query \
  'yql=select * from product where {targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)' \
  'approximate_query_string=blue t-shirt' \
  'input.query(q_embedding)=embed(@approximate_query_string)' \
  'ranking.profile=closeness_productname_description'
```

**Components:**
- **`nearestNeighbor(ProductName_embedding, q_embedding)`**: ANN search operator
- **`targetHits:100`**: Number of candidates to retrieve (higher = more accurate)
- **`approximate_query_string`**: Query text to embed
- **`input.query(q_embedding)`: `embed(@approximate_query_string)`**: Embed query text
- **`ranking.profile`**: Rank profile using `closeness()`

### Understanding targetHits

**`targetHits`** controls how many candidates are retrieved for ranking:
- **Higher value**: More accurate results, slower queries
- **Lower value**: Faster queries, may miss relevant results

**Recommendation:**
- Set `targetHits` to 10x your desired results
- Example: Want 10 results → Set `targetHits:100`

---

## Embedding Query Text

### Using embed() Function

The `embed()` function converts query text to embeddings using the configured embedder.

**Basic syntax:**
```json
{
    "approximate_query_string": "blue t-shirt",
    "input.query(q_embedding)": "embed(@approximate_query_string)"
}
```

**How it works:**
1. Query text "blue t-shirt" is sent to embedder
2. Embedder generates 384-dimensional vector
3. Vector is stored in `q_embedding` query tensor
4. Used in `nearestNeighbor` and `closeness()` ranking

### Alternative: Using @query

You can also use `@query` if you provide the query text separately:

```json
{
    "query": "blue t-shirt",
    "input.query(q_embedding)": "embed(@query)"
}
```

**Note:** `@query` is used when combining with lexical search (see below).

### Embedding Multiple Query Strings

You can embed different text for different purposes:

```json
{
    "approximate_query_string": "blue t-shirt",
    "query": "blue t-shirt",
    "input.query(q_embedding)": "embed(@approximate_query_string)",
    "input.query(q_embedding2)": "embed(@query)"
}
```

**Use case:** Different embeddings for different fields or ranking strategies.

---

## Multiple Embedding Fields

### OR Query (Union)

Search multiple embedding fields and combine results:

```http
POST https://<mTLS_ENDPOINT_DNS_GOES_HERE>/search/
Content-Type: application/json

{
    "yql": "select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR ({targetHits:100}nearestNeighbor(Description_embedding,q_embedding))",
    "ranking.profile": "closeness_productname_description",
    "approximate_query_string": "blue t-shirt",
    "input.query(q_embedding)": "embed(@approximate_query_string)"
}
```

**How it works:**
1. Searches both `ProductName_embedding` and `Description_embedding`
2. Combines results (union)
3. Rank profile combines similarities from both fields

### AND Query (Intersection)

Require matches in multiple fields:

```yql
select * from product where 
  {targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding) AND
  {targetHits:100}nearestNeighbor(Description_embedding,q_embedding)
```

**Use when:**
- Need strong matches in both fields
- Higher precision requirement

**Note:** AND may return fewer results than OR.

---

## Combining Lexical and Vector Search

### Hybrid Search (OR)

Combine lexical and vector search:

```http
POST https://<mTLS_ENDPOINT_DNS_GOES_HERE>/search/
Content-Type: application/json

{
    "yql": "select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()",
    "query": "blue t-shirt",
    "input.query(q_embedding)": "embed(@query)",
    "ranking.profile": "hybrid"
}
```

**How it works:**
1. `nearestNeighbor`: Vector search finds semantically similar documents
2. `userQuery()`: Lexical search finds keyword matches
3. Results are combined (union)
4. Hybrid rank profile combines both signals

### Hybrid Rank Profile

```vespa
rank-profile hybrid {
    inputs {
        query(q_embedding) tensor<float>(x[384])
    }

    function lexical_score() {
        expression: bm25(ProductName) + bm25(Description)
    }

    function semantic_score() {
        expression: closeness(field, ProductName_embedding) + 
                    closeness(field, Description_embedding)
    }

    first-phase {
        expression: lexical_score() * 0.5 + semantic_score() * 0.5
    }
}
```

**Benefits:**
- Combines strengths of both approaches
- Better recall (finds more relevant results)
- Better precision (more accurate results)

---

## Query Parameters

### hits

Control number of results returned:

```json
{
    "yql": "select * from product where {targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)",
    "hits": 10
}
```

**Note:** `hits` is the number of results returned, `targetHits` is the number of candidates for ranking.

### ranking.profile

Select rank profile:

```json
{
    "ranking.profile": "closeness_productname_description"
}
```

**Available profiles:**
- `default`: Lexical ranking (from Chapter 1)
- `closeness_productname_description`: Vector similarity ranking
- `hybrid`: Combined lexical + semantic (if defined)

### timeout

Set query timeout:

```json
{
    "timeout": "5s"
}
```

**Use when:**
- Queries may be slow
- Want to limit wait time

---

## Evaluation Framework

### Overview

The evaluation framework helps measure and compare search quality objectively.

**Components:**
- **`queries.csv`**: Test queries
- **`create_judgements.py`**: Generate relevance judgements
- **`evaluate.py`**: Compute metrics
- **`judgements.csv`**: Relevance judgements (generated)

### Step 1: Prepare Queries

Create `queries.csv`:
```csv
query_id,query_text
q1,blue t-shirt
q2,comfortable running shoes
q3,home decor items
```

### Step 2: Generate Judgements

**Setup:**
```bash
cd evaluation
cp env.example .env
# Edit .env with your configuration
```

**Configuration (`.env`):**
```bash
VESPA_ENDPOINT=https://your-endpoint.vespa-app.cloud
VESPA_CERT_PATH=/path/to/cert.pem
VESPA_KEY_PATH=/path/to/key.pem
OPENAI_API_KEY=your-key
HITS=20
```

**Run:**
```bash
pip install -r requirements.txt
python create_judgements.py
```

**How it works:**
1. Executes queries against Vespa (vector + lexical search)
2. Gets top results for each query
3. Uses OpenAI to judge relevance (0-3 scale)
4. Saves judgements to `judgements.csv`

**Judgement format:**
```csv
query_id,document_id,rating
q1,product123,3
q1,product456,2
q1,product789,1
```

**Rating scale:**
- **3**: Excellent match - directly answers the query
- **2**: Good match - relevant and useful
- **1**: Possible match - could be relevant for some users
- **0**: Irrelevant - does not answer the query

### Step 3: Evaluate Search Quality

**Configure query function in `evaluate.py`:**
```python
QUERY_FUNCTION = vector_search  # or lexical_search, hybrid_search
```

**Run:**
```bash
python evaluate.py
```

**Metrics computed:**
- **NDCG@10**: Normalized Discounted Cumulative Gain (primary metric)
- **MRR**: Mean Reciprocal Rank
- **Recall@10**: Recall at 10 results

**Output:**
```
Primary metric: ndcg_10
All results: {
    'ndcg_10': 0.75,
    'mrr': 0.82,
    'recall_10': 0.65
}
```

### Comparing Strategies

**Test different approaches:**
1. Run with `vector_search` → Semantic search metrics
2. Run with `lexical_search` → Lexical search metrics
3. Run with `hybrid_search` → Hybrid search metrics
4. Compare metrics to see which performs better

**Example workflow:**
```bash
# Test vector search
sed -i 's/QUERY_FUNCTION = .*/QUERY_FUNCTION = vector_search/' evaluate.py
python evaluate.py > results_vector.txt

# Test lexical search
sed -i 's/QUERY_FUNCTION = .*/QUERY_FUNCTION = lexical_search/' evaluate.py
python evaluate.py > results_lexical.txt

# Compare results
diff results_vector.txt results_lexical.txt
```

---

## Common Patterns

### Pattern 1: Simple Vector Search

```json
{
    "yql": "select * from product where {targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)",
    "approximate_query_string": "blue t-shirt",
    "input.query(q_embedding)": "embed(@approximate_query_string)",
    "ranking.profile": "closeness_productname_description"
}
```

**Use when:**
- Simple semantic search
- Single embedding field
- Fast queries needed

### Pattern 2: Multi-Field Vector Search

```json
{
    "yql": "select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR ({targetHits:100}nearestNeighbor(Description_embedding,q_embedding))",
    "approximate_query_string": "blue t-shirt",
    "input.query(q_embedding)": "embed(@approximate_query_string)",
    "ranking.profile": "closeness_productname_description"
}
```

**Use when:**
- Need to search multiple fields
- Want better recall
- Fields have different importance

### Pattern 3: Hybrid Search

```json
{
    "yql": "select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) OR userQuery()",
    "query": "blue t-shirt",
    "input.query(q_embedding)": "embed(@query)",
    "ranking.profile": "hybrid"
}
```

**Use when:**
- Want best of both worlds
- Need to handle exact matches and semantic similarity
- Production use case

### Pattern 4: Vector Search with Filters

```json
{
    "yql": "select * from product where ({targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)) AND Price < 1000",
    "approximate_query_string": "blue t-shirt",
    "input.query(q_embedding)": "embed(@approximate_query_string)",
    "ranking.profile": "closeness_productname_description"
}
```

**Use when:**
- Need to filter results
- Combine semantic search with business logic
- Constrain result set

### Pattern 5: Debugging with Summary Features

```json
{
    "yql": "select * from product where {targetHits:100}nearestNeighbor(ProductName_embedding,q_embedding)",
    "approximate_query_string": "blue t-shirt",
    "input.query(q_embedding)": "embed(@approximate_query_string)",
    "ranking.profile": "closeness_productname_description"
}
```

**Check response for `summary-features`:**
```json
{
    "root": {
        "children": [
            {
                "fields": {...},
                "relevance": 0.95,
                "summary-features": {
                    "closeness_productname": 0.92,
                    "closeness_description": 0.88
                }
            }
        ]
    }
}
```

**Use when:**
- Debugging ranking issues
- Understanding score contributions
- Tuning weights

---

## Troubleshooting

### nearestNeighbor Not Found

**Error**: `nearestNeighbor` operator not recognized

**Solutions:**
1. Ensure embedding field has `index` mode: `indexing: ... | index`
2. Check field type is `tensor<float>(x[384])` (or appropriate dimensions)
3. Verify `distance-metric` is set in attribute configuration
4. Redeploy after schema changes: `vespa deploy --wait 900`

### Query Embedding Not Generated

**Error**: Query embedding is null or query fails

**Solutions:**
1. Verify `embed()` function syntax: `embed(@approximate_query_string)`
2. Check embedder component is deployed in `services.xml`
3. Ensure query text is provided in `approximate_query_string`
4. Verify embedder can process the query text
5. Check embedder logs for errors

### No Results

**Issue**: Query returns no results

**Solutions:**
1. Check if data is fed: `vespa query 'yql=select * from product where true'`
2. Verify embeddings were generated during indexing
3. Check `targetHits` is not too low
4. Try broader query (remove filters)
5. Verify embedding fields are not null

### Low Similarity Scores

**Issue**: All results have very low relevance scores

**Solutions:**
1. Check embeddings are normalized (if using cosine similarity)
2. Verify `distance-metric` matches embedder normalization
3. Ensure query embedding is generated correctly
4. Check that document embeddings were generated during indexing
5. Verify embedding dimensions match (query vs. document)

### Performance Issues

**Issue**: Vector search is slow

**Solutions:**
1. Reduce `targetHits` (trade accuracy for speed)
2. Use fewer embedding fields in ranking
3. Consider two-phase ranking
4. Check ANN index configuration
5. Verify embeddings are stored as attributes (fast access)

### Evaluation Script Errors

**Error**: OpenAI API errors or connection issues

**Solutions:**
1. Verify `OPENAI_API_KEY` is set correctly
2. Check Vespa endpoint and certificates are configured
3. Ensure `queries.csv` and `judgements.csv` exist
4. Check Python dependencies are installed: `pip install -r requirements.txt`
5. Verify network connectivity to OpenAI API

---

## Additional Resources

- [Vespa Query API](https://docs.vespa.ai/en/query-api.html)
- [nearestNeighbor Reference](https://docs.vespa.ai/en/reference/query-language-reference.html#nearestneighbor)
- [YQL Reference](https://docs.vespa.ai/en/reference/query-language-reference.html)
- [Ranking Documentation](docs/RANKING.md)
- [LLM-as-a-Judge Guide](LLM_AS_JUDGE.md) – Comprehensive guide to using LLM-as-a-Judge
- [LLM-as-a-Judge Blog Post](https://blog.vespa.ai/improving-retrieval-with-llm-as-a-judge/)
