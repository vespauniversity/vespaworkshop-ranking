# Schema Design for Ranking

This document explains the schema design decisions for the e-commerce ranking tutorial.

**References:**
- [Vespa Schema Reference](https://docs.vespa.ai/en/schemas.html)
- [Field Types](https://docs.vespa.ai/en/reference/schema-reference.html#field)
- [Tensor Types](https://docs.vespa.ai/en/reference/tensor.html)

---

## Table of Contents

1. [Schema Overview](#schema-overview)
2. [Field Types and Indexing](#field-types-and-indexing)
3. [Text Fields](#text-fields)
4. [Attribute Fields](#attribute-fields)
5. [Tensor Fields](#tensor-fields)
6. [Rank Profiles](#rank-profiles)
7. [Design Decisions](#design-decisions)
8. [Common Issues](#common-issues)

---

## Schema Overview

The product schema is designed to support:
- **Text search** on ProductName and Description
- **Filtering** by brand, gender, price, color
- **Ranking** using text relevance, ratings, and user preferences
- **Business logic** integration (ratings, features)

**Full Schema:**
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

    fieldset default {
        fields: ProductName, Description
    }
}
```

---

## Field Types and Indexing

### Indexing Modes

Vespa supports three indexing modes:

1. **`index`**: Full-text indexed for search (enables `userQuery()`, BM25, nativeRank)
2. **`attribute`**: In-memory storage for fast filtering, sorting, grouping, ranking
3. **`summary`**: Stored and returned in query results (for display)

**Fields can have multiple modes:**
```vespa
field ProductName type string {
    indexing: summary | index  # Returned in results + searchable
}
```

### When to Use Each Mode

**`index`**: Use for text fields you want to search
- ProductName, Description
- Enables: `userQuery()`, `contains`, BM25, nativeRank

**`attribute`**: Use for fields you want to filter, sort, or use in ranking
- Price, Gender, Brand, AverageRating
- Enables: `Price < 1000`, `attribute(AverageRating)`, sorting

**`summary`**: Use for fields you want returned in results
- All fields that should appear in search results
- Required for fields to be returned (unless using custom summaries)

---

## Text Fields

### ProductName

```vespa
field ProductName type string {
    indexing: summary | index
}
```

**Purpose:**
- Primary search field (product titles)
- Used in ranking (BM25, nativeRank)

**Indexing:**
- `index`: Enables text search and ranking
- `summary`: Returned in query results

**Usage in Ranking:**
```vespa
bm25(ProductName)
nativeRank(ProductName)
```

### Description

```vespa
field Description type string {
    indexing: summary | index
}
```

**Purpose:**
- Secondary search field (product descriptions)
- Used in ranking (often weighted differently than title)

**Indexing:**
- `index`: Enables text search and ranking
- `summary`: Returned in query results

**Usage in Ranking:**
```vespa
bm25(Description)
nativeRank(Description) * 1.7  # Weighted higher than title
```

### Fieldset

```vespa
fieldset default {
    fields: ProductName, Description
}
```

**Purpose:**
- Groups fields for `userQuery()` searches
- Searches both fields simultaneously

**Usage:**
```bash
vespa query 'yql=select * from product where userQuery()' 'query=blue jeans'
# Searches both ProductName and Description
```

---

## Attribute Fields

### ProductBrand

```vespa
field ProductBrand type string {
    indexing: summary | attribute
}
```

**Purpose:**
- Filtering by brand
- Faceting (count products by brand)
- Used in ProductFeatures tensor

**Indexing:**
- `attribute`: Fast filtering and grouping
- `summary`: Returned in results

**Usage:**
```bash
vespa query 'yql=select * from product where ProductBrand = "Nike"'
```

### Gender

```vespa
field Gender type string {
    indexing: summary | attribute
}
```

**Purpose:**
- Filtering by gender category
- Used in ProductFeatures tensor for personalization

**Usage:**
```bash
vespa query 'yql=select * from product where Gender = "Women"'
```

### Price

```vespa
field Price type int {
    indexing: summary | attribute
}
```

**Purpose:**
- Filtering by price range
- Sorting by price
- Used in ProductFeatures tensor (PriceFactor)

**Usage:**
```bash
vespa query 'yql=select * from product where Price < 1000'
vespa query 'yql=select * from product where Price >= 500 and Price <= 2000'
```

**Note:** Price is `int` type. If you have decimal prices, use `float` or `double`.

### PrimaryColor

```vespa
field PrimaryColor type string {
    indexing: summary | attribute
}
```

**Purpose:**
- Filtering by color
- Faceting
- Used in ProductFeatures tensor

**Usage:**
```bash
vespa query 'yql=select * from product where PrimaryColor = "Blue"'
```

### AverageRating

```vespa
field AverageRating type string {
    indexing: summary | attribute
}
```

**Purpose:**
- Used in ranking (rating boost)
- Filtering by rating

**Why `string` instead of `float`?**
- The tutorial uses `string` for simplicity
- In production, use `float` or `double` for numeric operations
- `attribute(AverageRating)` works with string, but numeric is better

**Usage in Ranking:**
```vespa
first-phase {
    expression: bm25(ProductName) * attribute(AverageRating)
}
```

**Recommendation:** Change to `float` in production:
```vespa
field AverageRating type float {
    indexing: summary | attribute
}
```

---

## Tensor Fields

### ProductFeatures

```vespa
field ProductFeatures type tensor<float>(features{}) {
    indexing: summary | attribute
}
```

**Purpose:**
- Sparse tensor storing product features
- Used in personalized ranking (tensor dot product with user preferences)

**Tensor Structure:**
- **Type**: `tensor<float>(features{})`
  - `float`: Value type
  - `features{}`: Single mapped dimension (sparse tensor)
- **Format**: Sparse tensor with string keys

**Example Data:**
```json
{
  "ProductBrandDKNY": 1,
  "GenderUnisex": 1,
  "PrimaryColorBlack": 1,
  "PriceFactor": 0.93
}
```

**How It's Generated:**
The `enhance_data.py` script creates ProductFeatures:
- Categorical fields: `ProductBrand{Value}`, `Gender{Value}`, `PrimaryColor{Value}` → value: 1
- PriceFactor: `5 - log10(price)` → lower prices get higher factors

**Usage in Ranking:**
```vespa
second-phase {
    rerank-count: 10
    expression: sum(query(user_preferences) * attribute(ProductFeatures))
}
```

**Query Input Format:**
```json
{
  "ranking.features.query(user_preferences)": "{{features:GenderWomen}:1,{features:PriceFactor}:3}"
}
```

### Tensor Operations

**Dot Product:**
```vespa
sum(query(user_preferences) * attribute(ProductFeatures))
```

**How it works:**
1. Multiply matching features: `query(user_prefs) * ProductFeatures`
2. Sum all products: `sum(...)`
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

---

## Rank Profiles

Rank profiles are defined in separate `.profile` files in `app/schemas/product/`.

### Profile File Structure

```
app/schemas/product/
├── product.sd              # Schema definition
├── default.profile         # Default rank profile
├── bm25.profile            # BM25 ranking
├── nativeRankBM25.profile  # Combined signals
├── ratingboost.profile     # Business logic
└── preferences.profile    # Two-phase with preferences
```

### Profile File Format

Each `.profile` file contains a rank profile:

```vespa
rank-profile profile-name {
    # Optional: query inputs
    inputs {
        query(user_preferences) tensor<float>(features{})
    }
    
    # Optional: functions
    function my_function() {
        expression: bm25(ProductName)
    }
    
    # Optional: summary features
    summary-features: my_function
    
    # Required: first-phase
    first-phase {
        expression: my_function()
    }
    
    # Optional: second-phase
    second-phase {
        rerank-count: 10
        expression: expensive_operation()
    }
}
```

### Profile Inheritance

Profiles can inherit from other profiles:

```vespa
rank-profile advanced inherits default {
    first-phase {
        expression: nativeRank(ProductName, Description) * attribute(AverageRating)
    }
}
```

**Note:** This tutorial doesn't use inheritance, but it's available for more complex scenarios.

---

## Design Decisions

### Why Separate Profile Files?

**Benefits:**
- **Organization**: Each profile in its own file
- **Clarity**: Easy to see all profiles
- **Maintainability**: Change one profile without affecting others

**Alternative:** Define all profiles in `product.sd`:
```vespa
schema product {
    # ... fields ...
    
    rank-profile default { ... }
    rank-profile bm25 { ... }
    rank-profile ratingboost { ... }
}
```

**Tutorial Choice:** Separate files for learning clarity.

### Why AverageRating is String?

**Tutorial Decision:**
- Simplifies data preparation
- Works for basic ranking examples
- `attribute(AverageRating)` works with string

**Production Recommendation:**
- Use `float` or `double` for numeric operations
- Better for filtering: `AverageRating > 4.0`
- More efficient for numeric calculations

### Why Sparse Tensor for ProductFeatures?

**Benefits:**
- **Efficiency**: Only stores non-zero values
- **Flexibility**: Easy to add new features
- **Scalability**: Works with many features

**Alternative:** Separate attribute fields:
```vespa
field GenderWomen type int { ... }
field PriceFactor type float { ... }
```

**Why Tensor is Better:**
- Single field for all features
- Easy tensor operations (dot product)
- Flexible feature set (add new features without schema changes)

---

## Common Issues

### Field Not Found in Ranking

**Error**: Field used in ranking expression doesn't exist

**Solution:**
1. Verify field name matches exactly (case-sensitive)
2. Check field is defined in schema
3. Ensure field has `attribute` indexing if using `attribute()`

### Tensor Type Mismatch

**Error**: Tensor operations fail due to type mismatch

**Solution:**
1. Verify tensor types match: `tensor<float>(features{})`
2. Check query input format matches tensor structure
3. Ensure data format is correct (sparse tensor format)

### Indexing Mode Issues

**Error**: Can't use field in ranking (e.g., `bm25()` on non-indexed field)

**Solution:**
1. Text ranking requires `index` mode: `indexing: summary | index`
2. Attribute access requires `attribute` mode: `indexing: summary | attribute`
3. Check field indexing matches usage

### Profile Not Found

**Error**: Rank profile not recognized

**Solution:**
1. Verify profile file exists in `app/schemas/product/`
2. Check profile name matches exactly
3. Redeploy after adding/modifying profiles: `vespa deploy --wait 900`

---

## Additional Resources

- [Vespa Schema Reference](https://docs.vespa.ai/en/schemas.html)
- [Field Types](https://docs.vespa.ai/en/reference/schema-reference.html#field)
- [Tensor Types](https://docs.vespa.ai/en/reference/tensor.html)
- [Rank Profiles](docs/RANKING.md)
