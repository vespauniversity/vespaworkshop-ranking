# Semantic Ranking in Vespa - Detailed Guide

This document provides detailed explanations of semantic/vector ranking concepts used in the semantic e-commerce ranking tutorial.

**References:**
- [Vespa Embedding Documentation](https://docs.vespa.ai/en/embedding.html)
- [closeness() Reference](https://docs.vespa.ai/en/reference/ranking-expressions.html#closeness)
- [nearestNeighbor Reference](https://docs.vespa.ai/en/reference/query-language-reference.html#nearestneighbor)
- [Distance Metrics](https://docs.vespa.ai/en/reference/schema-reference.html#distance-metric)

---

## Table of Contents

1. [What is Semantic Ranking?](#what-is-semantic-ranking)
2. [Embeddings and Vector Similarity](#embeddings-and-vector-similarity)
3. [closeness() Function](#closeness-function)
4. [Distance Metrics](#distance-metrics)
5. [Rank Profiles for Vector Search](#rank-profiles-for-vector-search)
6. [Combining Multiple Embedding Fields](#combining-multiple-embedding-fields)
7. [Lexical vs. Semantic Ranking](#lexical-vs-semantic-ranking)
8. [Common Patterns](#common-patterns)
9. [Troubleshooting](#troubleshooting)

---

## What is Semantic Ranking?

**Semantic ranking** scores documents based on **semantic similarity** (meaning) rather than exact keyword matches. It uses vector embeddings to capture semantic relationships between queries and documents.

### Semantic vs. Lexical Ranking

**Lexical Ranking** (Chapter 1):
- Scores based on keyword matches
- Uses term frequency, document frequency
- Examples: BM25, nativeRank

**Semantic Ranking** (Chapter 2):
- Scores based on meaning similarity
- Uses vector embeddings and similarity metrics
- Examples: closeness(), cosine similarity

**Example:**
```
Query: "comfortable running shoes"

Lexical Search:
  - Matches: "running shoes" (exact keywords) → High score
  - Misses: "athletic footwear for jogging" (no exact match) → Low score

Semantic Search:
  - Matches: "running shoes" → High similarity
  - Matches: "athletic footwear for jogging" → High similarity (semantically similar)
  - Misses: "blue jeans" → Low similarity (different topic)
```

### Why Use Semantic Ranking?

**Benefits:**
- ✅ Handles synonyms and related terms
- ✅ Better for natural language queries
- ✅ Finds documents by meaning, not keywords
- ✅ Works across languages (with multilingual models)

**Limitations:**
- ❌ Requires embedding model and computation
- ❌ May miss exact keyword matches
- ❌ Can be slower than lexical search
- ❌ Quality depends on embedding model

**Best Practice:** Combine lexical + semantic (hybrid search) for best results.

---

## Embeddings and Vector Similarity

### What are Embeddings?

**Embeddings** are dense vector representations of text that capture semantic meaning. They're arrays of numbers (e.g., 384 dimensions) where similar texts have similar vectors.

**Properties:**
- **Fixed size**: All embeddings have the same dimensions
- **Dense**: Every dimension carries information
- **Semantic**: Similar meanings → Similar vectors

**Example:**
```
Text: "blue jeans"
Embedding: [0.1, 0.3, 0.7, -0.2, 0.5, ...] (384 numbers)

Text: "denim pants"
Embedding: [0.12, 0.28, 0.72, -0.18, 0.52, ...] (very similar)

Text: "red shirt"
Embedding: [0.8, 0.2, 0.1, 0.9, -0.3, ...] (different)
```

### How Embeddings are Generated

1. **Text Input**: "blue jeans"
2. **Tokenization**: Split into tokens
3. **Model Processing**: Neural network processes tokens
4. **Vector Output**: 384-dimensional vector

**In Vespa:**
- Embeddings are generated automatically during indexing
- Uses embedder component (e.g., Hugging Face model)
- Stored in tensor fields

### Vector Similarity

**Similarity** measures how close two vectors are in vector space.

**Common Metrics:**
- **Cosine Similarity**: Measures angle between vectors (0-1, higher = more similar)
- **Dot Product**: Sum of element-wise products (for normalized vectors)
- **Euclidean Distance**: Straight-line distance (lower = more similar)

**Example:**
```
Query vector:    [0.1, 0.3, 0.7, ...]
Document vector: [0.12, 0.28, 0.72, ...]

Cosine Similarity: 0.98 (very similar)
Euclidean Distance: 0.05 (very close)
```

**In Vespa:**
- `closeness()` function computes similarity
- Distance metric configured in schema
- Higher closeness = more similar = higher rank

---

## closeness() Function

The **closeness()** function computes vector similarity between query and document embeddings.

### Basic Syntax

```vespa
closeness(field, embedding_field)
```

**Parameters:**
- **`field`**: Query tensor (usually `field` keyword)
- **`embedding_field`**: Document embedding field

**Returns:** Similarity score (higher = more similar)

### Example

```vespa
rank-profile vector_search {
    inputs {
        query(q_embedding) tensor<float>(x[384])
    }

    function similarity() {
        expression: closeness(field, ProductName_embedding)
    }

    first-phase {
        expression: similarity()
    }
}
```

**How it works:**
1. Query vector `q_embedding` is compared to `ProductName_embedding`
2. `closeness()` computes similarity using configured distance metric
3. Returns score (e.g., 0.95 for very similar, 0.2 for different)

### Query Input

The query embedding must be provided as a query input:

```vespa
inputs {
    query(q_embedding) tensor<float>(x[384])
}
```

**In queries:**
```json
{
    "input.query(q_embedding)": "embed(@approximate_query_string)"
}
```

### Multiple Embedding Fields

You can compute closeness for multiple fields:

```vespa
function similarity_name() {
    expression: closeness(field, ProductName_embedding)
}

function similarity_desc() {
    expression: closeness(field, Description_embedding)
}

first-phase {
    expression: similarity_name() + similarity_desc()
}
```

**Best Practice:** Combine similarities from multiple fields for better ranking.

---

## Distance Metrics

**Distance metrics** determine how vector similarity is computed. The metric must match your embedding model's training.

### prenormalized-angular (Cosine Similarity)

**Use when:**
- Embeddings are normalized (unit length)
- Model was trained for cosine similarity
- Most common for modern embedding models

**Configuration:**
```vespa
field ProductName_embedding type tensor<float>(x[384]) {
    attribute {
        distance-metric: prenormalized-angular
    }
}
```

**How it works:**
- Computes dot product of normalized vectors
- Equivalent to cosine similarity
- Range: -1 to 1 (higher = more similar)

### euclidean

**Use when:**
- Model was trained for Euclidean distance
- Less common for text embeddings

**Configuration:**
```vespa
attribute {
    distance-metric: euclidean
}
```

**How it works:**
- Computes straight-line distance
- Lower distance = more similar
- Range: 0 to infinity (lower = more similar)

### innerproduct

**Use when:**
- Embeddings are not normalized
- Model was trained for inner product

**Configuration:**
```vespa
attribute {
    distance-metric: innerproduct
}
```

**How it works:**
- Computes dot product
- Higher product = more similar
- Range: -infinity to infinity (higher = more similar)

### Choosing the Right Metric

**For Arctic Embed (this tutorial):**
- Model produces normalized embeddings
- Use `prenormalized-angular` (cosine similarity)

**For other models:**
- Check model documentation
- Match metric to training objective
- Test different metrics to find best performance

---

## Rank Profiles for Vector Search

### Basic Vector Ranking Profile

```vespa
rank-profile vector_search {
    inputs {
        query(q_embedding) tensor<float>(x[384])
    }

    first-phase {
        expression: closeness(field, ProductName_embedding)
    }
}
```

**Components:**
- **`inputs`**: Define query tensor input
- **`closeness()`**: Compute similarity
- **`first-phase`**: Rank by similarity

### Combining Multiple Fields

```vespa
rank-profile multi_field {
    inputs {
        query(q_embedding) tensor<float>(x[384])
    }

    function sim_name() {
        expression: closeness(field, ProductName_embedding)
    }

    function sim_desc() {
        expression: closeness(field, Description_embedding)
    }

    first-phase {
        expression: sim_name() + sim_desc()
    }

    summary-features: sim_name sim_desc
}
```

**Benefits:**
- Searches multiple fields
- Combines similarities
- Exposes individual scores for debugging

### Weighted Combination

```vespa
first-phase {
    expression: sim_name() * 2.0 + sim_desc() * 1.0
}
```

**Why weight:**
- ProductName may be more important than Description
- Adjust weights based on evaluation results

### Two-Phase Vector Ranking

```vespa
rank-profile two_phase_vector {
    inputs {
        query(q_embedding) tensor<float>(x[384])
    }

    first-phase {
        expression: closeness(field, ProductName_embedding)
    }

    second-phase {
        rerank-count: 10
        expression: closeness(field, ProductName_embedding) + 
                    closeness(field, Description_embedding) +
                    attribute(AverageRating) * 0.5
    }
}
```

**Use when:**
- Second-phase features are expensive
- Need to combine vector similarity with other signals
- Want to optimize performance

---

## Combining Multiple Embedding Fields

### Why Combine Fields?

Different fields capture different aspects:
- **ProductName**: Product title, key information
- **Description**: Detailed information, context

Combining them improves recall and relevance.

### Simple Sum

```vespa
first-phase {
    expression: closeness(field, ProductName_embedding) + 
                closeness(field, Description_embedding)
}
```

**Pros:**
- Simple and fast
- Works well when fields are equally important

**Cons:**
- No field weighting
- May favor documents with high scores in both fields

### Weighted Sum

```vespa
first-phase {
    expression: closeness(field, ProductName_embedding) * 2.0 + 
                closeness(field, Description_embedding) * 1.0
}
```

**Pros:**
- Emphasizes important fields
- Tunable based on evaluation

**Cons:**
- Requires tuning weights
- May over-emphasize one field

### Maximum

```vespa
first-phase {
    expression: max(closeness(field, ProductName_embedding),
                   closeness(field, Description_embedding))
}
```

**Use when:**
- Want to match if ANY field is similar
- Good for recall (finds more results)

### Average

```vespa
first-phase {
    expression: (closeness(field, ProductName_embedding) + 
                 closeness(field, Description_embedding)) / 2.0
}
```

**Use when:**
- Want balanced matching across fields
- Good for precision (more relevant results)

---

## Lexical vs. Semantic Ranking

### Comparison

| Aspect | Lexical (BM25/nativeRank) | Semantic (closeness) |
|--------|---------------------------|----------------------|
| **Matching** | Exact keywords | Semantic meaning |
| **Synonyms** | Misses synonyms | Handles synonyms |
| **Natural Language** | Limited | Better |
| **Speed** | Very fast | Fast (with ANN) |
| **Exact Matches** | Excellent | Good |
| **Semantic Similarity** | Poor | Excellent |

### When to Use Each

**Use Lexical when:**
- Exact keyword matching is important
- Need very fast queries
- Queries are specific terms
- Documents have structured text

**Use Semantic when:**
- Natural language queries
- Need to handle synonyms
- Queries are conversational
- Documents have varied wording

**Best Practice:** Use **hybrid search** (combine both) for best results.

### Hybrid Ranking Example

```vespa
rank-profile hybrid {
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
- Better recall and precision
- Handles both exact matches and semantic similarity

---

## Common Patterns

### Pattern 1: Single Field Vector Search

```vespa
rank-profile single_field {
    inputs {
        query(q_embedding) tensor<float>(x[384])
    }

    first-phase {
        expression: closeness(field, ProductName_embedding)
    }
}
```

**Use when:**
- Only one field is important
- Simple use case
- Fast queries needed

### Pattern 2: Multi-Field with Weights

```vespa
rank-profile weighted_multi {
    inputs {
        query(q_embedding) tensor<float>(x[384])
    }

    function sim_name() {
        expression: closeness(field, ProductName_embedding)
    }

    function sim_desc() {
        expression: closeness(field, Description_embedding)
    }

    first-phase {
        expression: sim_name() * 2.0 + sim_desc()
    }
}
```

**Use when:**
- Multiple fields with different importance
- Need to tune field weights
- Want to emphasize certain fields

### Pattern 3: Vector + Business Logic

```vespa
rank-profile vector_business {
    inputs {
        query(q_embedding) tensor<float>(x[384])
    }

    function semantic_score() {
        expression: closeness(field, ProductName_embedding) + 
                    closeness(field, Description_embedding)
    }

    first-phase {
        expression: semantic_score() * attribute(AverageRating)
    }
}
```

**Use when:**
- Need to combine semantic relevance with business signals
- Want to boost highly-rated products
- Real-world ranking scenarios

### Pattern 4: Debugging with Summary Features

```vespa
rank-profile debug_vector {
    inputs {
        query(q_embedding) tensor<float>(x[384])
    }

    function sim_name() {
        expression: closeness(field, ProductName_embedding)
    }

    function sim_desc() {
        expression: closeness(field, Description_embedding)
    }

    summary-features: sim_name sim_desc

    first-phase {
        expression: sim_name() + sim_desc()
    }
}
```

**Use when:**
- Debugging ranking issues
- Understanding score contributions
- Tuning weights

---

## Troubleshooting

### Low Similarity Scores

**Issue**: All `closeness()` scores are very low (< 0.1)

**Solutions:**
1. Check embeddings are normalized (if using cosine similarity)
2. Verify `distance-metric` matches embedder normalization
3. Ensure query embedding is generated correctly
4. Check that document embeddings were generated during indexing
5. Verify embedding dimensions match (query vs. document)

### Inconsistent Results

**Issue**: Same query returns different results

**Solutions:**
1. Check `targetHits` is set appropriately (higher = more accurate)
2. Verify ANN index is built correctly
3. Ensure embeddings are generated consistently
4. Check for data quality issues

### Embedding Field Empty

**Issue**: Embedding fields are null or empty

**Solutions:**
1. Verify embedder component is configured correctly
2. Check `embed arctic` references correct component ID
3. Ensure source field (e.g., `ProductName`) has data
4. Re-feed documents after fixing schema
5. Check embedder logs for errors

### Query Embedding Not Generated

**Issue**: Query embedding is null or query fails

**Solutions:**
1. Verify `embed()` function syntax: `embed(@approximate_query_string)`
2. Check embedder component is deployed
3. Ensure query text is provided in `approximate_query_string`
4. Verify embedder can process the query text

### Performance Issues

**Issue**: Vector search is slow

**Solutions:**
1. Reduce `targetHits` (trade accuracy for speed)
2. Use fewer embedding fields in ranking
3. Consider two-phase ranking
4. Check ANN index configuration
5. Verify embeddings are stored as attributes (fast access)

---

## Additional Resources

- [Vespa Embedding Documentation](https://docs.vespa.ai/en/embedding.html)
- [closeness() Reference](https://docs.vespa.ai/en/reference/ranking-expressions.html#closeness)
- [nearestNeighbor Reference](https://docs.vespa.ai/en/reference/query-language-reference.html#nearestneighbor)
- [Distance Metrics](https://docs.vespa.ai/en/reference/schema-reference.html#distance-metric)
- [Rank Profiles](SCHEMA.md#rank-profiles)
