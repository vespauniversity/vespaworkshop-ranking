# Ranking in Vespa - Detailed Guide

This document provides detailed explanations of ranking concepts used in the e-commerce ranking tutorial.

**References:**
- [Vespa Ranking Documentation](https://docs.vespa.ai/en/ranking.html)
- [BM25 Reference](https://docs.vespa.ai/en/ranking/bm25.html)
- [nativeRank Reference](https://docs.vespa.ai/en/reference/nativerank.html)
- [Rank Profiles Reference](https://docs.vespa.ai/en/reference/schema-reference.html#rank-profile)

---

## Table of Contents

1. [What is Ranking?](#what-is-ranking)
2. [Rank Profiles](#rank-profiles)
3. [Ranking Phases](#ranking-phases)
4. [Ranking Functions](#ranking-functions)
5. [Functions and Summary Features](#functions-and-summary-features)
6. [Business Logic Integration](#business-logic-integration)
7. [Two-Phase Ranking](#two-phase-ranking)
8. [Query Inputs](#query-inputs)
9. [Tensor Operations](#tensor-operations)
10. [Common Patterns](#common-patterns)
11. [Troubleshooting](#troubleshooting)

---

## What is Ranking?

**Ranking** determines how relevant each document is to a query and sorts results by relevance score. It's the "scoring system" that decides which documents appear first in search results.

### Ranking vs. Matching

- **Matching**: Determines which documents match the query (boolean filter)
- **Ranking**: Scores and sorts matching documents by relevance

**Example:**
```
Query: "blue jeans"

Matching: Finds all documents containing "blue" AND "jeans"
Ranking: Scores each match:
  - Document A: "Blue denim jeans" → Score: 8.5
  - Document B: "Jeans in blue color" → Score: 7.2
  - Document C: "Blue shirt, jeans available" → Score: 5.1

Results: [A, B, C] (sorted by score, highest first)
```

### Why Ranking Matters

Without ranking, results are arbitrary. With ranking:
- Most relevant results appear first
- Users find what they're looking for faster
- Business goals can be incorporated (boost popular items, ratings, etc.)

---

## Rank Profiles

A **rank profile** defines how documents are scored. You can have multiple rank profiles in one schema for different use cases or A/B testing.

### Basic Structure

```vespa
schema product {
    document product {
        field ProductName type string {
            indexing: summary | index
        }
        field Description type string {
            indexing: summary | index
        }
        field AverageRating type float {
            indexing: summary | attribute
        }
    }
    
    rank-profile default {
        first-phase {
            expression: nativeRank(ProductName, Description)
        }
    }
}
```

### Multiple Rank Profiles

```vespa
rank-profile default {
    first-phase {
        expression: nativeRank(ProductName, Description)
    }
}

rank-profile bm25 {
    first-phase {
        expression: bm25(ProductName) + bm25(Description)
    }
}

rank-profile ratingboost {
    first-phase {
        expression: nativeRank(ProductName, Description) * attribute(AverageRating)
    }
}
```

### Using Rank Profiles

```bash
# Use default profile
vespa query 'yql=select * from product where userQuery()'

# Use specific profile
vespa query \
  'yql=select * from product where userQuery()' \
  'ranking.profile=bm25'

# Via HTTP
{
  "yql": "select * from product where userQuery()",
  "ranking.profile": "ratingboost"
}
```

---

## Ranking Phases

Vespa supports multiple ranking phases for performance optimization.

### First-Phase Ranking

**First-phase** ranking runs on **all matching documents** using fast, efficient features.

**Characteristics:**
- Fast (must be efficient for all matches)
- Runs on every matching document
- Typically uses: text relevance, simple attributes, basic math

**Example:**
```vespa
rank-profile default {
    first-phase {
        expression: bm25(ProductName) + bm25(Description)
    }
}
```

### Second-Phase Ranking

**Second-phase** ranking runs on **top N documents** from first phase using expensive features.

**Characteristics:**
- Can be expensive (only runs on top N)
- Only evaluates documents that passed first phase
- Typically uses: ML models, complex tensor operations, external APIs

**Example:**
```vespa
rank-profile two-phase {
    first-phase {
        expression: bm25(ProductName) * attribute(AverageRating)
    }
    second-phase {
        rerank-count: 10
        expression: sum(query(user_preferences) * attribute(ProductFeatures))
    }
}
```

**How it works:**
1. First phase scores all matches (e.g., 1000 documents)
2. Top 10 documents (by first-phase score) enter second phase
3. Second phase reranks those 10 using expensive features
4. Final results are the reranked top 10

### When to Use Two-Phase

**Use two-phase when:**
- ✅ You have expensive features (ML models, complex calculations)
- ✅ You need to balance performance and accuracy
- ✅ First-phase can effectively filter to top candidates

**Don't use two-phase when:**
- ❌ All features are fast (no performance benefit)
- ❌ You need exact ranking on all matches
- ❌ Second-phase features are critical for all documents

---

## Ranking Functions

Vespa provides built-in ranking functions for common use cases.

### nativeRank

**nativeRank** is Vespa's default text ranking algorithm, optimized for general text search.

**Features:**
- Term frequency (how often terms appear)
- Term proximity (how close terms are to each other)
- Field importance (can weight different fields)
- Document length normalization

**Syntax:**
```vespa
nativeRank(field1, field2, ...)
```

**Example:**
```vespa
rank-profile default {
    first-phase {
        expression: nativeRank(ProductName, Description)
    }
}
```

**When to use:**
- ✅ Default text search
- ✅ Need term proximity (phrases score higher)
- ✅ Want Vespa-optimized performance
- ✅ Simple use cases

### BM25

**BM25** (Best Matching 25) is an industry-standard ranking algorithm widely used in search systems.

**Features:**
- Term frequency (TF)
- Inverse document frequency (IDF)
- Field length normalization
- Tunable parameters (k1, b)

**Syntax:**
```vespa
bm25(field)
```

**Example:**
```vespa
rank-profile bm25 {
    first-phase {
        expression: bm25(ProductName) + bm25(Description)
    }
}
```

**When to use:**
- ✅ Want industry-standard algorithm
- ✅ Comparing with other search systems
- ✅ Need proven, well-tested algorithm
- ✅ Standard text search requirements

### nativeRank vs BM25

| Feature | nativeRank | BM25 |
|---------|------------|------|
| **Term Proximity** | ✅ Yes | ❌ No |
| **Industry Standard** | ❌ Vespa-specific | ✅ Widely used |
| **Performance** | ✅ Optimized for Vespa | ✅ Fast |
| **Tunability** | Limited | More parameters |
| **Use Case** | General text search | Standard text search |

**Recommendation:**
- Start with **nativeRank** for simplicity
- Use **BM25** if you need industry standard or are comparing systems
- Both work well for most text search scenarios

### Other Ranking Functions

**closeness**: Vector similarity (for embeddings)
```vespa
closeness(field, query_embedding)
```

**distance**: Vector distance (for embeddings)
```vespa
distance(field, query_embedding)
```

**attribute**: Access document attributes
```vespa
attribute(Price)
attribute(AverageRating)
```

**query**: Access query inputs
```vespa
query(user_preferences)
query(boost_factor)
```

---

## Functions and Summary Features

### Functions

**Functions** organize ranking expressions for reusability and readability.

**Basic Function:**
```vespa
rank-profile combined {
    function my_bm25() {
        expression: bm25(ProductName)
    }
    
    function my_nativeRank() {
        expression: nativeRank(Description) * 1.7
    }
    
    first-phase {
        expression: my_bm25() + my_nativeRank()
    }
}
```

**Benefits:**
- **Reusability**: Use same expression multiple times
- **Readability**: Name complex expressions
- **Debugging**: Can expose as summary-features

### Summary Features

**summary-features** expose intermediate scores in query results for debugging.

**Example:**
```vespa
rank-profile debug {
    function my_bm25() {
        expression: bm25(ProductName)
    }
    
    function my_nativeRank() {
        expression: nativeRank(Description) * 1.7
    }
    
    summary-features: my_bm25 my_nativeRank
    
    first-phase {
        expression: my_bm25() + my_nativeRank()
    }
}
```

**Query Result:**
```json
{
  "relevance": 12.5,
  "fields": {
    "ProductName": "Blue Jeans",
    "Description": "..."
  },
  "summary-features": {
    "my_bm25": 5.2,
    "my_nativeRank": 7.3
  }
}
```

**Use Cases:**
- Debugging ranking logic
- Understanding score contributions
- Tuning weights and parameters
- Monitoring ranking quality

---

## Business Logic Integration

Real-world ranking combines text relevance with business signals.

### Rating Boost

**Multiply** text relevance by rating to boost highly-rated products:

```vespa
rank-profile ratingboost {
    first-phase {
        expression: bm25(ProductName) * attribute(AverageRating)
    }
}
```

**Effect:**
- Product with rating 4.5 gets 4.5x boost
- Product with rating 2.0 gets 2.0x boost
- Higher ratings = higher final scores

### Additive Signals

**Add** signals instead of multiplying:

```vespa
rank-profile additive {
    first-phase {
        expression: bm25(ProductName) + attribute(AverageRating) * 0.5
    }
}
```

**Effect:**
- Text relevance + rating contribution
- More balanced (doesn't dominate with high ratings)
- Good when signals are on different scales

### Complex Business Logic

```vespa
rank-profile complex {
    function text_relevance() {
        expression: bm25(ProductName) + nativeRank(Description) * 1.5
    }
    
    function popularity_boost() {
        expression: log(attribute(ViewCount) + 1) * 0.3
    }
    
    function recency_boost() {
        expression: if(attribute(DaysSinceUpdate) < 7, 1.2, 1.0)
    }
    
    first-phase {
        expression: text_relevance() * attribute(AverageRating) + popularity_boost() * recency_boost()
    }
}
```

---

## Two-Phase Ranking

Two-phase ranking optimizes performance by using cheap features first, expensive features second.

### Basic Two-Phase

```vespa
rank-profile two-phase {
    first-phase {
        expression: bm25(ProductName) * attribute(AverageRating)
    }
    second-phase {
        rerank-count: 10
        expression: expensive_ml_model(query, document)
    }
}
```

### Performance Characteristics

**First Phase:**
- Runs on: All matching documents (e.g., 10,000)
- Features: Fast (text relevance, attributes)
- Time: ~10ms for 10,000 documents

**Second Phase:**
- Runs on: Top N from first phase (e.g., 10)
- Features: Expensive (ML models, complex calculations)
- Time: ~50ms for 10 documents

**Total Time:** ~60ms (vs 500ms if expensive features ran on all 10,000)

### Choosing rerank-count

**Too small (e.g., 5):**
- Risk missing relevant documents
- Second phase has limited candidates

**Too large (e.g., 1000):**
- Defeats performance purpose
- Second phase becomes expensive

**Recommended:**
- Start with 10-50
- Tune based on:
  - How many results users typically view
  - Performance requirements
  - Quality vs speed tradeoff

---

## Query Inputs

**Query inputs** allow dynamic ranking per query without changing the rank profile.

### Defining Query Inputs

```vespa
rank-profile personalized {
    inputs {
        query(user_preferences) tensor<float>(features{})
        query(boost_factor) double
    }
    
    first-phase {
        expression: bm25(ProductName) * query(boost_factor)
    }
}
```

### Using Query Inputs

**Via CLI:**
```bash
vespa query \
  'yql=select * from product where userQuery()' \
  'ranking.profile=personalized' \
  'ranking.features.query(boost_factor)=1.5'
```

**Via HTTP:**
```json
{
  "yql": "select * from product where userQuery()",
  "ranking.profile": "personalized",
  "ranking.features.query(boost_factor)": 1.5
}
```

### Common Use Cases

**User Preferences:**
```vespa
inputs {
    query(user_preferences) tensor<float>(features{})
}
```

**Dynamic Boosting:**
```vespa
inputs {
    query(category_boost) double
    query(brand_boost) double
}
```

**A/B Testing:**
```vespa
inputs {
    query(algorithm_version) string
}
```

---

## Tensor Operations

Tensors enable complex feature matching and personalization.

### Sparse Tensors

**ProductFeatures** is a sparse tensor:
```json
{
  "ProductBrandDKNY": 1,
  "GenderUnisex": 1,
  "PrimaryColorBlack": 1,
  "PriceFactor": 0.93
}
```

**Schema Definition:**
```vespa
field ProductFeatures type tensor<float>(features{}) {
    indexing: summary | attribute
}
```

### Tensor Dot Product

**Compute similarity** between query preferences and product features:

```vespa
rank-profile preferences {
    inputs {
        query(user_preferences) tensor<float>(features{})
    }
    
    second-phase {
        rerank-count: 10
        expression: sum(query(user_preferences) * attribute(ProductFeatures))
    }
}
```

**How it works:**
1. `query(user_preferences) * attribute(ProductFeatures)` multiplies matching features
2. `sum()` adds all matching feature products
3. Higher sum = better match to user preferences

**Example:**
```
User preferences: {GenderWomen: 1, PriceFactor: 3}
Product features: {GenderWomen: 1, PriceFactor: 0.93}

Calculation:
  GenderWomen: 1 * 1 = 1
  PriceFactor: 3 * 0.93 = 2.79
  Sum: 1 + 2.79 = 3.79
```

### Tensor Operations Reference

**Multiplication:**
```vespa
query(user_prefs) * attribute(ProductFeatures)
```

**Addition:**
```vespa
query(user_prefs) + attribute(ProductFeatures)
```

**Sum:**
```vespa
sum(query(user_prefs) * attribute(ProductFeatures))
```

**Reduce:**
```vespa
reduce(query(user_prefs) * attribute(ProductFeatures), sum, features)
```

---

## Common Patterns

### Pattern 1: Text + Business Signal

```vespa
rank-profile text_business {
    first-phase {
        expression: bm25(ProductName) * attribute(AverageRating)
    }
}
```

### Pattern 2: Multiple Fields Weighted

```vespa
rank-profile weighted {
    first-phase {
        expression: bm25(ProductName) * 2.0 + bm25(Description) * 1.0
    }
}
```

### Pattern 3: Conditional Boosting

```vespa
rank-profile conditional {
    first-phase {
        expression: bm25(ProductName) * if(attribute(Price) < 100, 1.5, 1.0)
    }
}
```

### Pattern 4: Logarithmic Scaling

```vespa
rank-profile log_scale {
    first-phase {
        expression: bm25(ProductName) * log(attribute(ViewCount) + 1)
    }
}
```

### Pattern 5: Two-Phase with Personalization

```vespa
rank-profile personalized_two_phase {
    inputs {
        query(user_preferences) tensor<float>(features{})
    }
    
    first-phase {
        expression: bm25(ProductName) * attribute(AverageRating)
    }
    
    second-phase {
        rerank-count: 10
        expression: sum(query(user_preferences) * attribute(ProductFeatures))
    }
}
```

---

## Troubleshooting

### Results Not Ranking Correctly

**Issue**: Results don't match expected ordering

**Solutions:**
1. Check `summary-features` to see intermediate scores
2. Verify fields are indexed/attribute as needed
3. Ensure data is fed correctly
4. Compare with solution files in `cheating/`

### Rank Profile Not Found

**Error**: `Unknown rank profile: bm25`

**Solutions:**
1. Ensure profile file exists in `app/schemas/product/`
2. Redeploy: `vespa deploy --wait 900`
3. Check profile name matches exactly (case-sensitive)
4. Verify profile syntax is correct

### Tensor Operation Errors

**Error**: Issues with tensor operations

**Solutions:**
1. Verify tensor field exists in schema
2. Check tensor structure matches query input
3. Ensure data includes tensor values
4. Verify tensor types match (e.g., `tensor<float>(features{})`)

### Query Input Not Working

**Error**: Query input not recognized

**Solutions:**
1. Ensure `inputs` section is defined in rank profile
2. Check tensor type matches query parameter
3. Verify query parameter format: `ranking.features.query(name)=value`
4. Check for typos in input name

### Performance Issues

**Issue**: Queries are slow

**Solutions:**
1. Use two-phase ranking for expensive features
2. Reduce `rerank-count` if second phase is slow
3. Optimize first-phase expression (avoid expensive operations)
4. Check that fields used in ranking are properly indexed

---

## Additional Resources

- [Vespa Ranking Documentation](https://docs.vespa.ai/en/ranking.html)
- [Rank Profiles Reference](https://docs.vespa.ai/en/reference/schema-reference.html#rank-profile)
- [BM25 Reference](https://docs.vespa.ai/en/ranking/bm25.html)
- [nativeRank Reference](https://docs.vespa.ai/en/reference/nativerank.html)
- [Tensor Operations](https://docs.vespa.ai/en/reference/tensor.html)
