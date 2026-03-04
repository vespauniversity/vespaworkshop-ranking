# Vespa Workshop – Ranking Series

This directory contains the Vespa Ranking Course with 5 progressive chapters covering lexical ranking through advanced recommendations.

## Project Content

### Chapter 1: Basic Lexical Ranking
**Location:** [`ecommerce_ranking_app/`](https://github.com/vespauniversity/vespaworkshop-ranking/blob/main/ecommerce_ranking_app/)
**📄 [README.md](https://github.com/vespauniversity/vespaworkshop-ranking/blob/main/ecommerce_ranking_app/README.md)**

**Concepts:**
- Rank profiles and ranking expressions
- BM25 and nativeRank for text relevance
- Combining multiple ranking signals (text + business logic)
- Functions and summary-features for debugging
- Two-phase ranking with user preference tensors

**Key Files:**
- `app/schemas/product/*.profile` - Rank profile files (default, bm25, nativeRankBM25, ratingboost, preferences)
- `dataset/products.jsonl` - Product data with ratings and features
- `solutions/` - Reference rank profiles
- `queries.http` - HTTP query examples for each step

---

### Chapter 2: Semantic / Vector Search
**Location:** [`semantic_ecommerce_ranking_app/`](https://github.com/vespauniversity/vespaworkshop-ranking/blob/main/semantic_ecommerce_ranking_app/)
**📄 [README.md](https://github.com/vespauniversity/vespaworkshop-ranking/blob/main/semantic_ecommerce_ranking_app/README.md)**

**Concepts:**
- Configuring embedder components (Hugging Face models) in `services.xml`
- Embedding fields with automatic vector generation
- `nearestNeighbor()` queries for approximate nearest neighbor (ANN) search
- `closeness()` ranking function for vector similarity
- Evaluation framework to measure search quality (NDCG)

**Key Files:**
- `app/schemas/product/*.profile` - Semantic rank profiles
- `dataset/` - Products without pre-computed embeddings
- `evaluation/` - Evaluation scripts and relevance judgements
- `solutions/` - Reference implementations
- `queries.http` - Semantic query examples

---

### Chapter 3: Hybrid Search + Learned Reranking
**Location:** [`hybrid_ecommerce_ranking_app/`](https://github.com/vespauniversity/vespaworkshop-ranking/blob/main/hybrid_ecommerce_ranking_app/)
**📄 [README.md](https://github.com/vespauniversity/vespaworkshop-ranking/blob/main/hybrid_ecommerce_ranking_app/README.md)**

**Concepts:**
- Combining lexical (BM25/nativeRank) and semantic (vector) search
- Hybrid queries using `userQuery()` and `nearestNeighbor()` operators
- Collecting rank features from search results for training data
- Training a LightGBM learned reranker
- Deploying ML models to Vespa for second-phase ranking
- Evaluating reranker performance with offline metrics

**Key Files:**
- `app/schemas/product/*.profile` - Hybrid and learned reranking profiles
- `dataset/` - Product data without embeddings
- `train_reranker/` - Scripts for collecting features and training LightGBM
- `solutions/` - Reference rank profiles
- `queries.http` - Hybrid query examples

---

### Chapter 4: Chunked Document Ranking (RAG)
**Location:** [`wiki_ranking_app/`](https://github.com/vespauniversity/vespaworkshop-ranking/blob/main/wiki_ranking_app/)
**📄 [README.md](https://github.com/vespauniversity/vespaworkshop-ranking/blob/main/wiki_ranking_app/README.md)**

**Concepts:**
- Array fields for storing document chunks
- Chunk embeddings with mapped tensors `tensor<float>(chunk{}, x[384])`
- Per-document ranking (treating all chunks as a single text)
- Per-chunk lexical ranking with BM25 on individual chunks
- Per-chunk semantic ranking using embeddings
- Hybrid ranking combining document-level and chunk-level signals
- Wikipedia metadata signals (links, freshness, revisions) for ranking

**Key Files:**
- `app/schemas/` - Wiki document schema with chunk fields
- `dataset/` - Wikipedia article data
- `assignments/` - Step-by-step rank profile tasks
- `solutions/` - Reference implementations
- `queries.http` - Chunked document query examples

---

### Chapter 5: Market Basket Analysis & Tensor Recommendations
**Location:** [`baskets_recommender_ranking_app/`](https://github.com/vespauniversity/vespaworkshop-ranking/blob/main/baskets_recommender_ranking_app/)
**📄 [README.md](https://github.com/vespauniversity/vespaworkshop-ranking/blob/main/baskets_recommender_ranking_app/README.md)**

**Concepts:**
- Vespa grouping API for market basket analysis
- Finding product co-occurrence patterns in transaction data
- Multi-dimensional tensors `tensor<int8>(category{}, features{})`
- User profile tensors with weighted preferences
- Tensor dot product for matching preferences to products
- Category-specific recommendations with filtering

**Key Files:**
- `baskets_app/` - Basket schema for co-occurrence analysis (Part 1)
- `category_recommender_app/` - Product schema with multi-dim tensors (Part 2)
- `dataset/basket_co-occurrence/groceries.jsonl` - Kaggle groceries dataset (9,835 baskets)
- `dataset/category_recommender/small_dataset_with_features.jsonl` - E-commerce products
- `solutions/` - Reference implementations
- `queries.http` - Grouping and recommendation query examples

---

## Course Progression

1. **Chapter 1** → Lexical ranking: rank profiles, BM25, business signals, two-phase ranking
2. **Chapter 2** → Semantic search: embeddings, vector similarity, evaluation
3. **Chapter 3** → Hybrid search: combine lexical + semantic, train learned reranker
4. **Chapter 4** → Chunked documents: RAG-style ranking with array fields and chunk embeddings
5. **Chapter 5** → Recommendations: basket analysis and tensor-based personalization

## Prerequisites

Each chapter builds on the previous. Before starting, ensure you have:
- Basic familiarity with Vespa schemas and deployment (from [Vespa Workshop 101](https://github.com/vespauniversity/vespaworkshop101))
- [Vespa CLI](https://docs.vespa.ai/en/clients/vespa-cli.html) installed (`brew install vespa-cli`)
- A Vespa Cloud account or local Docker deployment (8–12 GB RAM)

## Additional Resources

- [Vespa Documentation](https://docs.vespa.ai/)
- [Vespa Ranking Reference](https://docs.vespa.ai/en/ranking.html)
- [Vespa Sample Applications](https://github.com/vespa-engine/sample-apps)
- [Vespa University Repository](https://github.com/vespaai/university)
