# Schema Design for Hybrid Search

This document explains the schema design for Chapter 3: Hybrid Search with Learned Reranking.

## Table of Contents

- [Schema Overview](#schema-overview)
- [Document Fields](#document-fields)
- [Embedding Fields](#embedding-fields)
- [Embedder Components](#embedder-components)
- [Field Indexing Modes](#field-indexing-modes)
- [Rank Profile Organization](#rank-profile-organization)
- [Best Practices](#best-practices)

---

## Schema Overview

The `product` schema for hybrid search includes:

1. **Document fields**: ProductName, Description, Price, etc.
2. **Embedding fields**: ProductName_embedding, Description_embedding
3. **Attribute fields**: AverageRating, ProductFeatures for ranking
4. **Fieldset**: Default fieldset for lexical search

### Complete Schema

**File**: `app/schemas/product.sd`

```vespa
schema product {
    document product {
        field ProductID type string {
            indexing: summary
        }

        field ProductName type string {
            indexing: summary | index
        }

        field ProductBrand type string {
            indexing: summary | attribute
        }

        field Gender type string {
            indexing: summary | attribute
        }

        field Price type int {
            indexing: summary | attribute
        }

        field NumImages type int {
            indexing: summary | attribute
        }

        field Description type string {
            indexing: summary | index
        }

        field PrimaryColor type string {
            indexing: summary | attribute
        }

        field AverageRating type string {
            indexing: summary | attribute
        }

        field ProductFeatures type tensor<float>(features{}) {
            indexing: summary | attribute
        }
    }

    # Embedding fields (outside document block)
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

    # Fieldset for lexical search
    fieldset default {
        fields: ProductName, Description
    }
}
```

---

## Document Fields

### Text Fields (Lexical Search)

**ProductName**:
```vespa
field ProductName type string {
    indexing: summary | index
}
```

- **Purpose**: Product name for display and lexical search
- **Indexing modes**:
  - `summary`: Include in search results
  - `index`: Create inverted index for text search
- **Used in**: BM25, nativeRank, fieldMatch

**Description**:
```vespa
field Description type string {
    indexing: summary | index
}
```

- **Purpose**: Product description for display and lexical search
- **Indexing modes**: Same as ProductName
- **Used in**: BM25, nativeRank on description field

### Attribute Fields (Filtering and Ranking)

**ProductBrand**:
```vespa
field ProductBrand type string {
    indexing: summary | attribute
}
```

- **Purpose**: Brand name for filtering and grouping
- **Indexing mode**: `attribute` enables fast filtering
- **Used in**: Filters, grouping, potential ranking signal

**Gender**:
```vespa
field Gender type string {
    indexing: summary | attribute
}
```

- **Purpose**: Gender category (Men, Women, Unisex)
- **Indexing mode**: `attribute` for filtering
- **Used in**: Category filters

**Price**:
```vespa
field Price type int {
    indexing: summary | attribute
}
```

- **Purpose**: Product price in cents
- **Indexing mode**: `attribute` for range filters and ranking
- **Used in**: Price filters, ranking boost (price factor)

**AverageRating**:
```vespa
field AverageRating type string {
    indexing: summary | attribute
}
```

- **Purpose**: Average customer rating (1.0-5.0)
- **Indexing mode**: `attribute` for ranking
- **Used in**: Quality boost in ranking

**ProductFeatures**:
```vespa
field ProductFeatures type tensor<float>(features{}) {
    indexing: summary | attribute
}
```

- **Purpose**: Sparse tensor with product features
- **Type**: `tensor<float>(features{})` - sparse tensor with string keys
- **Example values**:
  ```json
  {
    "ProductBrandNike": 1.0,
    "GenderWomen": 1.0,
    "PrimaryColorBlue": 1.0,
    "PriceFactor": 3.2
  }
  ```
- **Used in**: User preference matching, feature-based ranking

---

## Embedding Fields

### Embedding Field Structure

Embedding fields are defined **outside the document block**:

```vespa
field ProductName_embedding type tensor<float>(x[384]) {
    indexing: input ProductName | embed arctic | attribute | index
    attribute {
        distance-metric: prenormalized-angular
    }
}
```

### Field Components

**1. Type declaration:**
```vespa
type tensor<float>(x[384])
```

- **`tensor<float>`**: Tensor with float values
- **`(x[384])`**: Dense tensor with 384 dimensions
- Dimension name `x` is convention for embeddings

**2. Indexing pipeline:**
```vespa
indexing: input ProductName | embed arctic | attribute | index
```

**Pipeline stages:**
- `input ProductName`: Take text from ProductName field
- `embed arctic`: Apply embedder component (configured in services.xml)
- `attribute`: Store for access in ranking
- `index`: Create HNSW index for ANN search

**3. Distance metric:**
```vespa
attribute {
    distance-metric: prenormalized-angular
}
```

**Distance metrics:**
- `prenormalized-angular`: Cosine similarity (for normalized embeddings)
- `angular`: Cosine similarity (Vespa normalizes)
- `euclidean`: L2 distance
- `dotproduct`: Dot product similarity
- `hamming`: Hamming distance (for binary embeddings)

**Why prenormalized-angular?**
- Arctic embedder produces normalized embeddings (via `<normalize>true</normalize>`)
- Cosine similarity on normalized vectors = dot product
- `prenormalized-angular` skips normalization step (faster)

### ProductName_embedding

```vespa
field ProductName_embedding type tensor<float>(x[384]) {
    indexing: input ProductName | embed arctic | attribute | index
    attribute {
        distance-metric: prenormalized-angular
    }
}
```

- **Source**: ProductName text field
- **Dimensions**: 384 (Arctic Embed XS model)
- **Purpose**: Semantic search on product names
- **Used in**: `nearestNeighbor(ProductName_embedding, q_embedding)`, `closeness(field, ProductName_embedding)`

### Description_embedding

```vespa
field Description_embedding type tensor<float>(x[384]) {
    indexing: input Description | embed arctic | attribute | index
    attribute {
        distance-metric: prenormalized-angular
    }
}
```

- **Source**: Description text field
- **Dimensions**: 384 (same as ProductName_embedding)
- **Purpose**: Semantic search on product descriptions
- **Used in**: `nearestNeighbor(Description_embedding, q_embedding)`, `closeness(field, Description_embedding)`

---

## Embedder Components

### Embedder Configuration

**File**: `app/services.xml`

```xml
<services version="1.0">
  <container id="default" version="1.0">
    <search />
    <document-api />

    <!-- Arctic Embedder Component -->
    <component id="arctic" type="hugging-face-embedder">
      <transformer-model url="https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/resolve/main/onnx/model.onnx"/>
      <tokenizer-model url="https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/raw/main/tokenizer.json"/>
      <normalize>true</normalize>
      <pooling-strategy>cls</pooling-strategy>
    </component>

  </container>

  <content id="content" version="1.0">
    <documents>
      <document type="product" mode="index"/>
    </documents>
    <nodes>
      <node distribution-key="0" hostalias="node1"/>
    </nodes>
  </content>
</services>
```

### Component Configuration Details

**Component ID:**
```xml
<component id="arctic" type="hugging-face-embedder">
```

- **`id="arctic"`**: Referenced in schema (`embed arctic`)
- **`type="hugging-face-embedder"`**: Vespa's Hugging Face embedder component

**Model URLs:**
```xml
<transformer-model url="https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/resolve/main/onnx/model.onnx"/>
<tokenizer-model url="https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/raw/main/tokenizer.json"/>
```

- **transformer-model**: ONNX format model for embeddings
- **tokenizer-model**: Tokenizer for text preprocessing
- Downloaded automatically on first deployment

**Normalization:**
```xml
<normalize>true</normalize>
```

- **Purpose**: Normalize embeddings to unit length
- **Effect**: Enables use of `prenormalized-angular` distance metric
- **Result**: Cosine similarity = dot product (faster computation)

**Pooling Strategy:**
```xml
<pooling-strategy>cls</pooling-strategy>
```

- **Purpose**: How to convert token embeddings to sentence embedding
- **Options**:
  - `cls`: Use CLS token (BERT-style models)
  - `mean`: Average all token embeddings
  - `max`: Max pooling

### Arctic Embed XS Model

**Model characteristics:**
- **Dimensions**: 384
- **Size**: ~50MB (ONNX format)
- **Max tokens**: 512 (longer texts truncated)
- **Training**: Optimized for retrieval tasks
- **Performance**: Good balance of quality and speed

**Alternatives:**
- Arctic Embed Small (768 dimensions, better quality)
- Arctic Embed Medium (1024 dimensions, best quality)
- Other Hugging Face models (e.g., all-MiniLM-L6-v2)

**Changing embedder:**
1. Update `services.xml` with new model URLs
2. Update schema embedding field dimensions (e.g., `x[768]`)
3. Redeploy
4. Re-feed all documents (embeddings regenerated)

---

## Field Indexing Modes

### Indexing Mode Summary

| Mode | Purpose | Access | Example |
|------|---------|--------|---------|
| `summary` | Include in results | Document retrieval | ProductName, Description |
| `index` | Text search | Query matching | ProductName, Description |
| `attribute` | Filtering, ranking | Direct access | Price, AverageRating |

### summary

```vespa
field ProductID type string {
    indexing: summary
}
```

- **Purpose**: Include field in search result summaries
- **Storage**: Document store
- **Access**: Retrieved when document is a search result
- **Use case**: Display-only fields (ProductID, ProductName)

### index

```vespa
field ProductName type string {
    indexing: summary | index
}
```

- **Purpose**: Enable text search on field
- **Storage**: Inverted index + document store
- **Access**: Query matching via `userQuery()`, `contains`, etc.
- **Use case**: Searchable text fields (ProductName, Description)

### attribute

```vespa
field Price type int {
    indexing: summary | attribute
}
```

- **Purpose**: Enable filtering, grouping, and ranking
- **Storage**: In-memory attribute store
- **Access**: Fast direct access (no index lookup)
- **Use case**: Filters, sorting, ranking (Price, AverageRating, Gender)

### Combinations

**summary | index** (Searchable text):
```vespa
field ProductName type string {
    indexing: summary | index
}
```

- Text search enabled
- Included in results
- Example: ProductName, Description

**summary | attribute** (Filterable):
```vespa
field Gender type string {
    indexing: summary | attribute
}
```

- Fast filtering
- Included in results
- Example: Gender, ProductBrand, Price

**summary | attribute | index** (Embedding fields):
```vespa
field ProductName_embedding type tensor<float>(x[384]) {
    indexing: input ProductName | embed arctic | attribute | index
    ...
}
```

- Attribute: Access in ranking
- Index: ANN search (HNSW)
- Example: Embedding fields

---

## Rank Profile Organization

### Rank Profile Files

Rank profiles can be:
1. **Inline** in schema file (simple profiles)
2. **Separate .profile files** (recommended for complex profiles)

**Directory structure:**
```
app/schemas/
├── product.sd
└── product/
    ├── default.profile
    ├── closeness_productname_description.profile
    ├── hybrid.profile
    └── rerank.profile
```

### Profile Inheritance

**Base profile** (hybrid.profile):
```vespa
rank-profile hybrid {
    function native_rank_name() {
        expression: nativeRank(ProductName)
    }

    function closeness_productname() {
        expression: closeness(field, ProductName_embedding)
    }

    first-phase {
        expression: native_rank_name() + closeness_productname()
    }
}
```

**Derived profile** (rerank.profile):
```vespa
rank-profile rerank inherits hybrid {
    second-phase {
        rerank-count: 20
        expression: lightgbm("lightgbm_model.json")
    }
}
```

**Benefits of inheritance:**
- Reuse functions and features
- Avoid code duplication
- Easy to add second-phase to existing profile

---

## Best Practices

### Field Design

**1. Use appropriate indexing modes:**
- Text search → `index`
- Filtering/ranking → `attribute`
- Display only → `summary`

**2. Minimize index fields:**
- Indexing is expensive (storage + processing)
- Only index fields that need text search
- Use attributes for structured data

**3. Choose correct field types:**
- Text: `string`
- Numbers: `int`, `long`, `float`, `double`
- Booleans: `byte` (0/1)
- Tensors: `tensor<float>(...)`

### Embedding Field Design

**1. Match embedder output dimensions:**
- Arctic XS: 384 dimensions
- Update schema if changing embedder

**2. Use appropriate distance metrics:**
- Normalized embeddings → `prenormalized-angular`
- Unnormalized embeddings → `angular` or `euclidean`
- Binary embeddings → `hamming`

**3. Create embeddings for searchable fields:**
- ProductName → ProductName_embedding
- Description → Description_embedding
- Don't embed non-text fields

**4. Consider embedding multiple fields:**
- Separate embeddings: Better for field-specific weighting
- Combined embedding: Simpler, fewer fields

### Embedder Configuration

**1. Choose appropriate model:**
- Smaller models (384d): Faster, less memory
- Larger models (768d+): Better quality, slower

**2. Enable normalization:**
```xml
<normalize>true</normalize>
```
- Enables faster distance computation
- Use with `prenormalized-angular` metric

**3. Model caching:**
- Vespa caches model in memory
- First query may be slower (model loading)
- Subsequent queries fast

### Schema Evolution

**Adding new fields:**
1. Add field to schema
2. Deploy schema update
3. Re-feed documents OR use partial updates

**Changing embedding dimensions:**
1. Update embedder configuration
2. Update embedding field type
3. Deploy
4. Re-feed ALL documents (embeddings regenerated)

**Removing fields:**
1. Remove from schema
2. Deploy
3. Old data remains but inaccessible
4. Optional: Re-feed to clean up

---

## Summary

**Key schema components:**
- **Document fields**: ProductName, Description (text search)
- **Attribute fields**: Price, AverageRating (filtering, ranking)
- **Embedding fields**: ProductName_embedding, Description_embedding (semantic search)
- **Embedder component**: Arctic Embed XS (configured in services.xml)

**Indexing modes:**
- `summary`: Include in results
- `index`: Enable text search
- `attribute`: Enable filtering and ranking

**Embedding configuration:**
- Type: `tensor<float>(x[384])`
- Pipeline: `input Field | embed embedder | attribute | index`
- Distance metric: `prenormalized-angular` (for normalized embeddings)

**Best practices:**
- Use appropriate indexing modes for each field
- Match embedding dimensions to embedder output
- Organize rank profiles in separate .profile files
- Use profile inheritance to avoid duplication

**Resources:**
- [Vespa Schema Reference](https://docs.vespa.ai/en/reference/schema-reference.html)
- [Embeddings Documentation](https://docs.vespa.ai/en/embedding.html)
- [Hugging Face Embedder](https://docs.vespa.ai/en/rag/embedding.html#huggingface-embedder)
- [Distance Metrics](https://docs.vespa.ai/en/content/attributes.html#distance-metric)
