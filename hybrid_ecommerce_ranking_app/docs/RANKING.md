# Hybrid Ranking Concepts

This document explains the ranking concepts for Chapter 3: Hybrid Search with Learned Reranking.

## Table of Contents

- [Hybrid Search Overview](#hybrid-search-overview)
- [Hybrid vs Lexical vs Semantic](#hybrid-vs-lexical-vs-semantic)
- [Hybrid Rank Profiles](#hybrid-rank-profiles)
- [Combining Ranking Signals](#combining-ranking-signals)
- [Learned Reranking](#learned-reranking)
- [Two-Phase Ranking](#two-phase-ranking)
- [Feature Engineering](#feature-engineering)
- [Best Practices](#best-practices)

---

## Hybrid Search Overview

**Hybrid search** combines multiple search paradigms to achieve better performance than any single approach:

### Search Paradigms

1. **Lexical Search** (keyword-based)
   - Uses inverted indexes
   - Matches terms exactly
   - Ranking: BM25, nativeRank, TF-IDF

2. **Semantic Search** (meaning-based)
   - Uses vector embeddings
   - Matches by semantic similarity
   - Ranking: cosine similarity, dot product

3. **Hybrid Search** (combines both)
   - Retrieves via OR operator (union)
   - Ranks using features from both paradigms
   - Best of both worlds

### Why Hybrid Works

Different query types benefit from different search methods:

**Navigational queries** (exact match intent):
- Query: "Nike Air Max 270"
- **Lexical wins**: Finds exact product name
- Semantic may find related but wrong products

**Informational queries** (broad intent):
- Query: "comfortable running shoes for marathon training"
- **Semantic wins**: Understands intent, finds relevant products
- Lexical requires exact keyword matches

**Hybrid approach**: Automatically adapts to query type by:
- Retrieving candidates from both methods
- Ranking with features from both
- Letting the best matches rise to the top

### Research Evidence

**MS MARCO Passage Ranking Leaderboard:**
- Top systems all use hybrid approaches
- Typical improvement: 10-20% NDCG over single-paradigm

**Vespa Blog Studies:**
- [Improving Zero-Shot Ranking](https://blog.vespa.ai/improving-zero-shot-ranking-with-vespa/)
- Hybrid retrieval + reranking: 15-25% NDCG improvement

---

## Hybrid vs Lexical vs Semantic

### Comparative Analysis

| Aspect | Lexical | Semantic | Hybrid |
|--------|---------|----------|--------|
| **Retrieval** | Inverted index | Vector index (HNSW) | Both |
| **Query Processing** | Tokenization | Embedding | Both |
| **Matching** | Exact terms | Semantic similarity | Both |
| **Ranking** | BM25, nativeRank | Closeness | Combined |
| **Best For** | Navigational, exact | Informational, broad | All query types |
| **Precision** | High (when matched) | Variable | High |
| **Recall** | Limited (keyword dependent) | High | Highest |
| **Speed** | Very fast | Fast | Fast |
| **Setup Complexity** | Low | Medium | Medium |

### When Each Excels

**Lexical search excels at:**
- Brand names: "Nike", "Adidas", "Apple"
- Product codes: "SKU-12345", "Model XYZ"
- Acronyms: "CPU", "GPU", "RAM"
- Exact phrases: "blue denim jeans"
- Rare terms: "quinoa", "macchiato"

**Semantic search excels at:**
- Natural language: "shoes good for running marathons"
- Synonyms: "pants" → "trousers"
- Paraphrases: "affordable" → "budget-friendly"
- Conceptual matches: "gift for dad" → relevant products
- Multi-lingual queries (with multi-lingual embeddings)

**Hybrid search excels at:**
- All of the above
- Mixed queries: "Nike running shoes for beginners"
  - Lexical: Captures "Nike" brand
  - Semantic: Understands "for beginners" intent
- Robust to query variation

### Performance Comparison

Typical NDCG@10 results (e-commerce search):

```
Lexical only:     0.65 - 0.70
Semantic only:    0.68 - 0.73
Hybrid:           0.75 - 0.80
Hybrid + Rerank:  0.80 - 0.85
```

**Key insight**: Hybrid provides 5-15% improvement over best single-paradigm approach.

---

## Hybrid Rank Profiles

### Basic Hybrid Ranking

A hybrid rank profile combines features from both lexical and semantic search:

```vespa
rank-profile hybrid {
    ## Lexical features
    function native_rank_name() {
        expression: nativeRank(ProductName)
    }
    function native_rank_description() {
        expression: nativeRank(Description)
    }

    ## Semantic features
    inputs {
        query(q_embedding) tensor<float>(x[384])
    }

    function closeness_productname() {
        expression: closeness(field, ProductName_embedding)
    }
    function closeness_description() {
        expression: closeness(field, Description_embedding)
    }

    ## Combine features
    first-phase {
        expression: native_rank_name() + native_rank_description() +
                    closeness_productname() + closeness_description()
    }
}
```

### Feature Functions

**Why use functions?**
- **Reusability**: Define once, use in multiple phases
- **Debugging**: Expose via `summary-features`
- **Clarity**: Self-documenting code
- **Modularity**: Easy to add/remove features

**Naming convention:**
- `native_rank_name` → nativeRank on ProductName
- `closeness_productname` → closeness on ProductName_embedding
- `AverageRating` → attribute(AverageRating)

### Query Inputs

Query inputs allow dynamic ranking parameters:

```vespa
inputs {
    query(q_embedding) tensor<float>(x[384])
}
```

**Usage in queries:**
```json
{
  "input.query(q_embedding)": "embed(@query)"
}
```

This embeds the query text and passes it to the rank profile.

---

## Combining Ranking Signals

### Combination Strategies

**1. Additive (simple sum):**
```vespa
expression: feature1() + feature2() + feature3()
```
- **Pros**: Simple, interpretable, no tuning needed
- **Cons**: Assumes equal importance, linear combination
- **Use case**: Baseline, quick experimentation

**2. Weighted sum:**
```vespa
expression: feature1() * 2.0 + feature2() * 1.5 + feature3()
```
- **Pros**: Control feature importance
- **Cons**: Requires manual tuning
- **Use case**: Domain knowledge about feature importance

**3. Multiplicative boost:**
```vespa
expression: (feature1() + feature2()) * attribute(AverageRating)
```
- **Pros**: Strong boosting effect
- **Cons**: Can dominate ranking if not normalized
- **Use case**: Business logic (ratings, popularity)

**4. Learned combination (LightGBM):**
```vespa
expression: lightgbm("model.json")
```
- **Pros**: Optimal weights, non-linear, handles interactions
- **Cons**: Requires training data, more complex
- **Use case**: Production systems with historical data

### Normalization

Features have different scales:
- `nativeRank`: 0 to ~1000
- `closeness`: 0 to 1
- `AverageRating`: 1 to 5
- `Price`: 100 to 100,000

**Normalization strategies:**

**1. Feature scaling:**
```vespa
function normalized_price() {
    expression: 1.0 / (1.0 + log(attribute(Price)))
}
```

**2. Min-max scaling:**
```vespa
function scaled_rating() {
    expression: (attribute(AverageRating) - 1.0) / 4.0
}
```

**3. Let ML do it:**
- LightGBM handles different scales automatically
- No manual normalization needed

### Feature Interactions

**Simple features** (first-order):
- `nativeRank(ProductName)`
- `closeness(field, ProductName_embedding)`
- `attribute(AverageRating)`

**Feature interactions** (second-order):
```vespa
function quality_boost() {
    expression: closeness_productname() * attribute(AverageRating)
}
```

**Complex interactions** (learned):
- LightGBM automatically discovers feature interactions
- No need to manually specify all combinations

---

## Learned Reranking

### Why Learned Reranking?

**Manual ranking challenges:**
- Hard to find optimal feature weights
- Can't capture non-linear relationships
- Misses feature interactions
- Time-consuming to tune

**Learned ranking benefits:**
- Automatically finds optimal weights
- Captures non-linear relationships
- Discovers feature interactions
- Optimizes for ranking metrics (NDCG)

### LightGBM for Ranking

**LightGBM** (Light Gradient Boosting Machine):
- Fast, efficient gradient boosting framework
- Optimized for ranking tasks
- Handles large datasets
- Built-in support for NDCG optimization

**How it works:**
1. **Training**: Learns from query-document pairs with relevance labels
2. **Model**: Ensemble of decision trees
3. **Prediction**: Scores documents for reranking
4. **Export**: JSON format compatible with Vespa

**Example model structure:**
```
LightGBM Model
├── Tree 1: Splits on closeness_productname, AverageRating
├── Tree 2: Splits on native_rank_name, Price
├── Tree 3: Splits on closeness_description, ...
└── Final score: Sum of all tree predictions
```

### Training Data Requirements

**Minimum requirements:**
- 100+ query-document pairs
- 20+ unique queries
- Diverse relevance grades (0-3)
- Representative query distribution

**Recommended:**
- 1000+ query-document pairs
- 100+ unique queries
- Balanced relevance distribution
- Multiple documents per query (5-20)

**Data quality matters more than quantity:**
- Accurate labels > many labels
- Diverse queries > similar queries
- Representative of production > historical bias

---

## Two-Phase Ranking

### Why Two-Phase?

**Challenge**: ML models are expensive to evaluate
- Complex computations per document
- Scales linearly with matched documents
- Can't evaluate 10,000+ documents per query

**Solution**: Two-phase ranking
1. **First-phase**: Fast ranking on all matches
2. **Second-phase**: Expensive ranking on top N

### Architecture

```vespa
rank-profile rerank inherits hybrid {
    ## First-phase: Fast hybrid ranking
    first-phase {
        expression: native_rank_name() + closeness_productname() + ...
    }

    ## Second-phase: ML model on top 20
    second-phase {
        rerank-count: 20
        expression: lightgbm("lightgbm_model.json")
    }
}
```

**How it works:**
1. Query matches 10,000 documents
2. First-phase scores all 10,000 (fast)
3. Top 20 from first-phase selected
4. Second-phase rescores top 20 (expensive)
5. Final results from second-phase

### Performance Analysis

**Example latency breakdown:**

| Phase | Documents | Time per Doc | Total Time |
|-------|-----------|--------------|------------|
| Match | 10,000 | 0.01ms | 100ms |
| First-phase | 10,000 | 0.1ms | 1,000ms |
| Second-phase | 20 | 5ms | 100ms |
| **Total** | | | **~1,200ms** |

Without two-phase (ML on all docs):
- 10,000 × 5ms = 50,000ms = **50 seconds** ❌

With two-phase:
- First: 1,000ms + Second: 100ms = **~1.2 seconds** ✓

**Rule of thumb:**
- `rerank-count`: 10-50 documents
- Higher count: Better quality, slower
- Lower count: Faster, may miss good results

### Choosing rerank-count

**Factors to consider:**
- **Query latency budget**: How much time do you have?
- **Result diversity**: More documents = more diversity
- **First-phase quality**: If first-phase is good, lower count OK
- **Model complexity**: More complex models → lower count

**Recommendations:**
- Start with 20
- Measure latency and quality
- Increase if quality improves significantly
- Decrease if latency too high

---

## Feature Engineering

### Essential Features for Hybrid Ranking

**Lexical features:**
```vespa
function native_rank_name() {
    expression: nativeRank(ProductName)
}
function bm25_name() {
    expression: bm25(ProductName)
}
function fieldMatch_name() {
    expression: fieldMatch(ProductName)
}
```

**Semantic features:**
```vespa
function closeness_productname() {
    expression: closeness(field, ProductName_embedding)
}
function embedding_similarity() {
    expression: sum(query(q_embedding) * attribute(ProductName_embedding))
}
```

**Document attributes:**
```vespa
function rating() {
    expression: attribute(AverageRating)
}
function price_factor() {
    expression: 5.0 - log10(attribute(Price))
}
```

**Hybrid features:**
```vespa
function quality_weighted_relevance() {
    expression: (native_rank_name() + closeness_productname()) * attribute(AverageRating)
}
```

### Advanced Features

**Field match features:**
```vespa
function term_proximity() {
    expression: fieldMatch(ProductName).proximity
}
function query_coverage() {
    expression: fieldMatch(ProductName).queryCompleteness
}
```

**Freshness features:**
```vespa
function recency_boost() {
    expression: 1.0 / (1.0 + (now - attribute(timestamp)) / 86400)
}
```

**Diversity features:**
```vespa
function brand_penalty() {
    expression: if(attribute(ProductBrand) == "common_brand", 0.9, 1.0)
}
```

### Feature Selection

**Good features have:**
- **Discriminative power**: Separates relevant from irrelevant
- **Low correlation**: Adds new information
- **Stable**: Works across queries
- **Efficient**: Fast to compute

**Feature importance analysis:**
After training LightGBM, review feature importance:
```
closeness_productname: 0.32  ← Most important
native_rank_name: 0.21
AverageRating: 0.15
closeness_description: 0.12
native_rank_description: 0.10
Price: 0.05
NumImages: 0.03             ← Low importance
PrimaryColor: 0.02          ← Consider removing
```

**Remove low-importance features:**
- Reduces model complexity
- Improves inference speed
- May prevent overfitting

---

## Best Practices

### Hybrid Search Best Practices

1. **Start simple, iterate**
   - Begin with basic additive hybrid ranking
   - Measure baseline performance
   - Add complexity only if needed

2. **Use OR operator for retrieval**
   ```yql
   where nearestNeighbor(...) OR userQuery()
   ```
   - Combines result sets
   - Each document scored by all features

3. **Set appropriate targetHits**
   - Too low: Miss good results
   - Too high: Slower, more false positives
   - Recommended: 100-200

4. **Expose features via summary-features**
   ```vespa
   summary-features: native_rank_name closeness_productname
   ```
   - Essential for debugging
   - Required for training data collection

### Reranking Best Practices

1. **Collect diverse training data**
   - Multiple query types
   - Balanced relevance grades
   - Representative of production traffic

2. **Use cross-validation**
   - Prevents overfitting
   - Provides reliable performance estimates
   - 5-fold is typical

3. **Monitor feature importance**
   - Validates feature engineering
   - Identifies redundant features
   - Guides future improvements

4. **A/B test in production**
   - Offline metrics don't always match online
   - Test reranker vs baseline
   - Measure business metrics (CTR, conversion)

5. **Retrain regularly**
   - User preferences change
   - Product catalog evolves
   - Quarterly retraining recommended

### Common Pitfalls

**1. Overfitting**
- **Symptom**: High train NDCG, low test NDCG
- **Solution**: More data, simpler model, regularization

**2. Data leakage**
- **Symptom**: Unrealistically high performance
- **Solution**: Ensure train/test split by query

**3. Feature scaling issues**
- **Symptom**: One feature dominates
- **Solution**: Normalize features or use LightGBM

**4. Insufficient training data**
- **Symptom**: Poor generalization
- **Solution**: Collect more diverse queries

**5. Ignoring first-phase quality**
- **Symptom**: Second-phase can't fix bad first-phase
- **Solution**: Optimize first-phase before adding second

### Performance Optimization

1. **First-phase optimization**
   - Keep expressions simple
   - Avoid expensive functions
   - Use attribute fields

2. **Second-phase optimization**
   - Low rerank-count (10-20)
   - Lightweight models
   - Consider model quantization

3. **Embedder optimization**
   - Use smaller models if possible (e.g., Arctic XS)
   - Cache embeddings
   - Batch embedding generation

4. **Query optimization**
   - Appropriate targetHits
   - Filter early (before ranking)
   - Use query profiles for common patterns

---

## Summary

**Key takeaways:**

1. **Hybrid search** combines lexical and semantic search for best overall performance
2. **Learned reranking** optimizes ranking using machine learning (LightGBM)
3. **Two-phase ranking** balances performance and quality
4. **Feature engineering** is critical for reranking success
5. **Evaluation** is essential to measure and improve search quality

**Next steps:**
- Implement hybrid ranking in your application
- Collect relevance judgements
- Train a reranker
- Measure improvement
- Iterate on features and model

**Resources:**
- [Vespa Ranking Documentation](https://docs.vespa.ai/en/ranking.html)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Learning to Rank Tutorial](https://docs.vespa.ai/en/learning-to-rank.html)
