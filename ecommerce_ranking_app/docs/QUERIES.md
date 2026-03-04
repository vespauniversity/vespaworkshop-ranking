# Query Guide for E-commerce Ranking

This document provides query examples and patterns for testing rank profiles in the e-commerce ranking tutorial.

**References:**
- [Vespa Query API](https://docs.vespa.ai/en/query-api.html)
- [YQL Reference](https://docs.vespa.ai/en/reference/query-language-reference.html)

---

## Table of Contents

1. [Basic Queries](#basic-queries)
2. [Using Rank Profiles](#using-rank-profiles)
3. [Query Inputs](#query-inputs)
4. [Field Selection](#field-selection)
5. [Filtering and Search](#filtering-and-search)
6. [Summary Features](#summary-features)
7. [Common Patterns](#common-patterns)
8. [Troubleshooting](#troubleshooting)

---

## Basic Queries

### Simple Search

**Find all products:**
```bash
vespa query 'yql=select * from product where true'
```

**Search by product name:**
```bash
vespa query 'yql=select * from product where ProductName contains "shirt"'
```

**Search in multiple fields:**
```bash
vespa query 'yql=select * from product where ProductName contains "shirt" OR Description contains "shirt"'
```

### HTTP Queries

**Basic search via HTTP:**
```http
POST https://<mTLS_ENDPOINT_DNS_GOES_HERE>/search/
Content-Type: application/json

{
  "yql": "select * from product where ProductName contains 'shirt'"
}
```

**Using userQuery() (searches fieldset):**
```http
POST https://<mTLS_ENDPOINT_DNS_GOES_HERE>/search/
Content-Type: application/json

{
  "yql": "select * from product where userQuery()",
  "query": "blue jeans"
}
```

---

## Using Rank Profiles

### Via CLI

**Default profile:**
```bash
vespa query 'yql=select * from product where ProductName contains "shirt"'
```

**Specific profile:**
```bash
vespa query \
  'yql=select * from product where ProductName contains "shirt"' \
  'ranking.profile=bm25'
```

**Multiple parameters:**
```bash
vespa query \
  'yql=select * from product where ProductName contains "shirt" OR Description contains "shirt"' \
  'ranking.profile=ratingboost' \
  'hits=10'
```

### Via HTTP

**Basic profile selection:**
```http
POST https://<mTLS_ENDPOINT_DNS_GOES_HERE>/search/
Content-Type: application/json

{
  "yql": "select * from product where ProductName contains 'shirt'",
  "ranking.profile": "bm25"
}
```

**With hits parameter:**
```http
POST https://<mTLS_ENDPOINT_DNS_GOES_HERE>/search/
Content-Type: application/json

{
  "yql": "select * from product where ProductName contains 'shirt'",
  "ranking.profile": "ratingboost",
  "hits": 5
}
```

---

## Query Inputs

Query inputs allow dynamic ranking per query.

### Basic Query Input

**Via CLI:**
```bash
vespa query \
  'yql=select * from product where ProductName contains "shirt"' \
  'ranking.profile=preferences' \
  'ranking.features.query(user_preferences)={{features:GenderWomen}:1,{features:PriceFactor}:2}'
```

**Via HTTP:**
```http
POST https://<mTLS_ENDPOINT_DNS_GOES_HERE>/search/
Content-Type: application/json

{
  "yql": "select * from product where ProductName contains 'shirt' OR Description contains 'shirt'",
  "ranking.profile": "preferences",
  "ranking.features.query(user_preferences)": "{{features:GenderWomen}:1,{features:GenderUnisex}:0.7,{features:PriceFactor}:3}"
}
```

### Query Input Examples

**Price-conscious user:**
```json
{
  "ranking.features.query(user_preferences)": "{{features:PriceFactor}:5}"
}
```

**Brand-loyal user:**
```json
{
  "ranking.features.query(user_preferences)": "{{features:ProductBrandNike}:2,{features:ProductBrandAdidas}:1.5}"
}
```

**Color-picky user:**
```json
{
  "ranking.features.query(user_preferences)": "{{features:PrimaryColorBlue}:2,{features:PrimaryColorBlack}:1.5}"
}
```

**Combined preferences:**
```json
{
  "ranking.features.query(user_preferences)": "{{features:GenderWomen}:1,{features:PrimaryColorBlue}:1.5,{features:PriceFactor}:2,{features:ProductBrandNike}:1.2}"
}
```

### Understanding Tensor Format

The tensor format for sparse tensors is:
```
{{dimension:key}:value,{dimension:key}:value,...}
```

**Example breakdown:**
```json
"{{features:GenderWomen}:1,{features:PriceFactor}:3}"
```

- `features` = dimension name
- `GenderWomen` = feature key
- `1` = feature value
- `PriceFactor` = another feature key
- `3` = its value

---

## Field Selection

### Select All Fields

```bash
vespa query 'yql=select * from product where ProductName contains "shirt"'
```

### Select Specific Fields

```bash
vespa query 'yql=select ProductID, ProductName, Price, AverageRating from product where ProductName contains "shirt"'
```

**Via HTTP:**
```http
POST https://<mTLS_ENDPOINT_DNS_GOES_HERE>/search/
Content-Type: application/json

{
  "yql": "select ProductID, ProductName, Price, AverageRating from product where ProductName contains 'shirt'"
}
```

### Why Select Specific Fields?

- **Reduce response size**: Only return needed fields
- **Performance**: Less data to transfer
- **Clarity**: Focus on relevant information

---

## Filtering and Search

### Combining Search and Filters

**Search with price filter:**
```bash
vespa query 'yql=select * from product where ProductName contains "shirt" and Price < 1000'
```

**Search with multiple filters:**
```bash
vespa query 'yql=select * from product where ProductName contains "shirt" and Price < 1000 and Gender = "Women"'
```

**Search with OR logic:**
```bash
vespa query 'yql=select * from product where (ProductName contains "shirt" OR Description contains "shirt") and Price < 1000'
```

### Filtering Examples

**By brand:**
```bash
vespa query 'yql=select * from product where ProductBrand = "Nike"'
```

**By color:**
```bash
vespa query 'yql=select * from product where PrimaryColor = "Blue"'
```

**By rating:**
```bash
vespa query 'yql=select * from product where AverageRating > 4.0'
```

**Price range:**
```bash
vespa query 'yql=select * from product where Price >= 500 and Price <= 2000'
```

---

## Summary Features

Summary features expose intermediate ranking scores for debugging.

### Viewing Summary Features

**Via CLI:**
```bash
vespa query \
  'yql=select * from product where ProductName contains "shirt"' \
  'ranking.profile=nativeRankBM25'
```

**Check the response** for `summary-features`:
```json
{
  "root": {
    "children": [
      {
        "fields": {
          "ProductName": "Blue Shirt",
          "Description": "..."
        },
        "relevance": 12.5,
        "summary-features": {
          "my_bm25": 5.2,
          "my_nativeRank": 7.3
        }
      }
    ]
  }
}
```

### Using Summary Features

**Understanding score contributions:**
- `my_bm25`: BM25 score for ProductName
- `my_nativeRank`: nativeRank score for Description (weighted 1.7x)
- `relevance`: Final combined score

**Debugging ranking:**
1. Check individual feature scores
2. Verify weights are applied correctly
3. Compare scores across different products
4. Tune weights based on feature contributions

---

## Common Patterns

### Pattern 1: Compare Rank Profiles

**Test multiple profiles on same query:**
```bash
# Default profile
vespa query 'yql=select * from product where ProductName contains "shirt"'

# BM25 profile
vespa query \
  'yql=select * from product where ProductName contains "shirt"' \
  'ranking.profile=bm25'

# Rating boost profile
vespa query \
  'yql=select * from product where ProductName contains "shirt" OR Description contains "shirt"' \
  'ranking.profile=ratingboost'
```

### Pattern 2: Test User Preferences

**Different user preference profiles:**
```bash
# Price-conscious user
vespa query \
  'yql=select * from product where ProductName contains "shirt"' \
  'ranking.profile=preferences' \
  'ranking.features.query(user_preferences)={{features:PriceFactor}:5}'

# Brand-loyal user
vespa query \
  'yql=select * from product where ProductName contains "shirt"' \
  'ranking.profile=preferences' \
  'ranking.features.query(user_preferences)={{features:ProductBrandNike}:3}'
```

### Pattern 3: Limit Results

**Top 5 results:**
```bash
vespa query \
  'yql=select * from product where ProductName contains "shirt" limit 5'
```

**Via hits parameter:**
```bash
vespa query \
  'yql=select * from product where ProductName contains "shirt"' \
  'hits=5'
```

### Pattern 4: Search with Filters

**Search + filter + ranking:**
```bash
vespa query \
  'yql=select * from product where ProductName contains "shirt" and Price < 1000 and AverageRating > 4.0' \
  'ranking.profile=ratingboost'
```

### Pattern 5: Field-Specific Search

**Search only in title:**
```bash
vespa query 'yql=select * from product where ProductName contains "shirt"'
```

**Search only in description:**
```bash
vespa query 'yql=select * from product where Description contains "shirt"'
```

**Search in both:**
```bash
vespa query 'yql=select * from product where ProductName contains "shirt" OR Description contains "shirt"'
```

---

## Troubleshooting

### No Results

**Issue**: Query returns no results

**Solutions:**
1. Check if data is fed: `vespa query 'yql=select * from product where true'`
2. Verify field names match schema
3. Check spelling of search terms
4. Try broader query (remove filters)

### Wrong Ranking

**Issue**: Results not ranked as expected

**Solutions:**
1. Check `summary-features` to see intermediate scores
2. Verify rank profile is being used: check `ranking.profile` parameter
3. Compare with solution files in `cheating/`
4. Ensure fields are indexed/attribute as needed

### Query Input Not Working

**Issue**: Query input not affecting ranking

**Solutions:**
1. Verify `inputs` section in rank profile
2. Check tensor format matches (sparse tensor format)
3. Ensure query parameter name matches: `ranking.features.query(name)`
4. Check for typos in feature keys

### Performance Issues

**Issue**: Queries are slow

**Solutions:**
1. Limit results: `hits=10` or `limit 10`
2. Add filters to reduce matching documents
3. Use two-phase ranking for expensive features
4. Check that fields are properly indexed

---

## Additional Resources

- [Vespa Query API](https://docs.vespa.ai/en/query-api.html)
- [YQL Reference](https://docs.vespa.ai/en/reference/query-language-reference.html)
- [Ranking Documentation](docs/RANKING.md)
