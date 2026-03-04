# Schema Design for Semantic Search

This document explains the schema design decisions for semantic/vector search in the semantic e-commerce ranking tutorial.

**References:**
- [Vespa Schema Reference](https://docs.vespa.ai/en/schemas.html)
- [Embedding Documentation](https://docs.vespa.ai/en/embedding.html)
- [Tensor Types](https://docs.vespa.ai/en/reference/tensor.html)
- [Hugging Face Embedder](https://docs.vespa.ai/en/embedding.html#huggingface-embedder)
- [Embeddings & Models Notes](EMBEDDINGS.md)

---

## Table of Contents

1. [Schema Overview](#schema-overview)
2. [Embedder Components](#embedder-components)
3. [Embedding Fields](#embedding-fields)
4. [Distance Metrics](#distance-metrics)
5. [Indexing Modes](#indexing-modes)
6. [Rank Profiles](#rank-profiles)
7. [Design Decisions](#design-decisions)
8. [Common Issues](#common-issues)

---

## Schema Overview

The product schema is designed to support:
- **Text search** on ProductName and Description (lexical)
- **Vector search** on embedding fields (semantic)
- **Filtering** by brand, gender, price, color
- **Ranking** using vector similarity (`closeness()`)
- **Hybrid search** combining lexical and semantic

**Full Schema:**
```vespa
schema product {
    document product {
        # Text fields (from Chapter 1)
        field ProductID type string {
            indexing: summary
        }
        field ProductName type string {
            indexing: summary | index
        }
        field Description type string {
            indexing: summary | index
        }
        
        # Attribute fields (from Chapter 1)
        field ProductBrand type string {
            indexing: summary | attribute
        }
        field Gender type string {
            indexing: summary | attribute
        }
        field Price type int {
            indexing: summary | attribute
        }
        field AverageRating type string {
            indexing: summary | attribute
        }
        # ... other fields ...
    }

    # Embedding fields (Chapter 2)
    field ProductName_embedding type tensor<float>(x[384]) {
        indexing: input ProductName | embed arctic | attribute | index
        attribute {
            distance-metric: prenormalized-angular
        }
    }

    field Description_embedding type tensor<float>(x[384]) {
        indexing: input Description | embed arctic | attribute | index
        attribute {
            distance-metric: prenormalized-angular
        }
    }

    fieldset default {
        fields: ProductName, Description
    }
}
```

---

## Embedder Components

### What is an Embedder?

An **embedder** is a Vespa component that converts text to embeddings (vectors). It's configured in `services.xml` and referenced in schema fields.

### Hugging Face Embedder

This tutorial uses the **Hugging Face embedder** with the **Snowflake Arctic Embed XS** model.

**Configuration in `services.xml`:**
```xml
<component id="arctic" type="hugging-face-embedder">
    <transformer-model url="https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/resolve/main/onnx/model.onnx"/>
    <tokenizer-model url="https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/raw/main/tokenizer.json"/>
    <normalize>true</normalize>
    <pooling-strategy>cls</pooling-strategy>
</component>
```

**Configuration Details:**
- **`id="arctic"`**: Component identifier (referenced in schema as `embed arctic`)
- **`type="hugging-face-embedder"`**: Uses Hugging Face models
- **`transformer-model`**: ONNX model URL for embeddings
- **`tokenizer-model`**: Tokenizer URL for text preprocessing
- **`normalize>true</normalize>`**: Normalize embeddings to unit length (enables cosine similarity)
- **`pooling-strategy>cls</pooling-strategy>`**: Use CLS token pooling

### Model Details

**Arctic Embed XS:**
- **Dimensions**: 384
- **Size**: ~50MB
- **Normalized**: Yes (unit length vectors)
- **Distance Metric**: Cosine similarity (prenormalized-angular)
- **Language**: Multilingual (supports many languages)

**Model Download:**
- Model is downloaded automatically on first use
- Cached locally after first download
- May take a few minutes on first deployment

### Other Embedder Types

Vespa supports other embedder types:
- **ONNX embedder**: Custom ONNX models
- **Cloud embedder**: External embedding services
- **Custom embedders**: Java-based embedders

**For this tutorial:** Hugging Face embedder is simplest and most common.

---

## Embedding Fields

### What are Embedding Fields?

**Embedding fields** automatically generate vectors from source text fields during indexing. They use the configured embedder to convert text to embeddings.

### Basic Structure

```vespa
field ProductName_embedding type tensor<float>(x[384]) {
    indexing: input ProductName | embed arctic | attribute | index
    attribute {
        distance-metric: prenormalized-angular
    }
}
```

**Components:**
- **`type tensor<float>(x[384])`**: 384-dimensional float vector
- **`input ProductName`**: Source field for embedding
- **`embed arctic`**: Use "arctic" embedder component
- **`attribute | index`**: Store as attribute and index for ANN search
- **`distance-metric`**: How to measure similarity

### How Embedding Fields Work

**During Indexing:**
1. Document is indexed with `ProductName: "blue jeans"`
2. `input ProductName` extracts text: "blue jeans"
3. `embed arctic` sends text to embedder component
4. Embedder generates 384-dimensional vector
5. Vector is stored in `ProductName_embedding` field
6. Field is indexed for fast ANN search

**During Querying:**
1. Query text is embedded: "blue t-shirt" → query vector
2. `nearestNeighbor` finds documents with nearest vectors
3. `closeness()` ranks by similarity

### Multiple Embedding Fields

You can create embeddings for multiple source fields:

```vespa
field ProductName_embedding type tensor<float>(x[384]) {
    indexing: input ProductName | embed arctic | attribute | index
    attribute {
        distance-metric: prenormalized-angular
    }
}

field Description_embedding type tensor<float>(x[384]) {
    indexing: input Description | embed arctic | attribute | index
    attribute {
        distance-metric: prenormalized-angular
    }
}
```

**Benefits:**
- Search multiple fields semantically
- Combine similarities for better ranking
- Different fields capture different aspects

### Embedding Dimensions

**Choosing dimensions:**
- **384**: Arctic Embed XS (this tutorial)
- **768**: Many BERT-based models
- **1536**: OpenAI text-embedding-ada-002
- **1024**: Many sentence transformers

**Important:** Query and document embeddings must have the same dimensions.

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
- Fast computation (dot product)

**For Arctic Embed:**
- Model produces normalized embeddings (`normalize=true`)
- Use `prenormalized-angular` (cosine similarity)

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

## Indexing Modes

### Embedding Field Indexing

Embedding fields need both `attribute` and `index` modes:

```vespa
field ProductName_embedding type tensor<float>(x[384]) {
    indexing: input ProductName | embed arctic | attribute | index
}
```

**Why both:**
- **`attribute`**: Fast access for ranking (`closeness()`)
- **`index`**: Enables `nearestNeighbor` queries (ANN search)

### Source Field Indexing

Source fields (e.g., `ProductName`) need `index` mode for lexical search:

```vespa
field ProductName type string {
    indexing: summary | index
}
```

**Why:**
- Enables lexical search (`userQuery()`, `contains`)
- Used for hybrid search (lexical + semantic)

---

## Rank Profiles

### Basic Vector Ranking Profile

```vespa
rank-profile closeness_productname_description {
    inputs {
        query(q_embedding) tensor<float>(x[384])
    }

    function closeness_productname() {
        expression: closeness(field, ProductName_embedding)
    }

    function closeness_description() {
        expression: closeness(field, Description_embedding)
    }

    first-phase {
        expression: closeness_productname() + closeness_description()
    }

    summary-features: closeness_productname closeness_description
}
```

**Components:**
- **`inputs`**: Define query tensor input
- **`closeness()`**: Compute similarity
- **`first-phase`**: Rank by similarity
- **`summary-features`**: Expose scores for debugging

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

**Important:** Query tensor type must match embedding field type.

---

## Design Decisions

### Why Separate Embedding Fields?

**Benefits:**
- **Automatic generation**: Embeddings created during indexing
- **Separation of concerns**: Text fields for lexical, embedding fields for semantic
- **Flexibility**: Can use different embedders for different fields
- **Performance**: Embeddings stored separately, optimized for ANN search

**Alternative:** Generate embeddings externally and store as tensors
- More control but more complexity
- Requires external embedding pipeline

**Tutorial Choice:** Automatic embedding fields for simplicity.

### Why Arctic Embed XS?

**Benefits:**
- **Small size**: ~50MB (fast download, low memory)
- **Fast**: Optimized for speed
- **Good quality**: Competitive with larger models
- **Multilingual**: Supports many languages
- **Normalized**: Enables cosine similarity

**Alternatives:**
- Larger models (better quality, slower)
- Specialized models (domain-specific, may require fine-tuning)

**Tutorial Choice:** Arctic Embed XS for balance of quality and speed.

### Why 384 Dimensions?

**Trade-offs:**
- **Lower dimensions** (128, 256): Faster, less memory, lower quality
- **Higher dimensions** (768, 1536): Slower, more memory, better quality

**384 dimensions:**
- Good balance for most use cases
- Fast enough for real-time search
- Good quality for semantic search

**For production:** Test different dimensions based on your requirements.

### Why prenormalized-angular?

**Benefits:**
- **Fast**: Dot product is very fast
- **Normalized**: Works well with normalized embeddings
- **Standard**: Most common for text embeddings

**For Arctic Embed:**
- Model produces normalized embeddings
- Cosine similarity is the right metric

---

## Common Issues

### Embedder Not Found

**Error**: `Unknown embedder: arctic`

**Solutions:**
1. Ensure embedder component is defined in `services.xml`
2. Check component ID matches schema reference (`embed arctic`)
3. Verify component is inside `<container>` section
4. Redeploy: `vespa deploy --wait 900`

### Embedding Field Not Generated

**Issue**: Embedding fields are empty or null

**Solutions:**
1. Verify `embed arctic` references correct component ID
2. Check embedder component is deployed correctly
3. Ensure source field (e.g., `ProductName`) has data
4. Re-feed documents after fixing schema
5. Check embedder logs for errors

### Dimension Mismatch

**Error**: Query and document embeddings have different dimensions

**Solutions:**
1. Verify embedding field type matches query input type
2. Check embedder produces correct dimensions
3. Ensure all embedding fields use same embedder
4. Verify query embedding is generated correctly

### Distance Metric Mismatch

**Issue**: Low similarity scores or incorrect ranking

**Solutions:**
1. Check `distance-metric` matches embedder normalization
2. Verify embedder `normalize` setting
3. For normalized embeddings, use `prenormalized-angular`
4. Test different metrics to find best performance

### Performance Issues

**Issue**: Slow indexing or queries

**Solutions:**
1. Check embedder model size (smaller = faster)
2. Verify embeddings are stored as attributes (fast access)
3. Consider reducing embedding dimensions
4. Check ANN index configuration
5. Monitor resource usage (CPU, memory)

### Embedding Quality Issues

**Issue**: Poor search results

**Solutions:**
1. Try different embedding models
2. Check if model supports your language/domain
3. Consider fine-tuning for domain-specific use
4. Test with evaluation framework
5. Compare with lexical search to identify issues

---

## Additional Resources

- [Vespa Schema Reference](https://docs.vespa.ai/en/schemas.html)
- [Embedding Documentation](https://docs.vespa.ai/en/embedding.html)
- [Hugging Face Embedder](https://docs.vespa.ai/en/embedding.html#huggingface-embedder)
- [Tensor Types](https://docs.vespa.ai/en/reference/tensor.html)
- [Distance Metrics](https://docs.vespa.ai/en/reference/schema-reference.html#distance-metric)
- [Rank Profiles](docs/RANKING.md#rank-profiles-for-vector-search)
