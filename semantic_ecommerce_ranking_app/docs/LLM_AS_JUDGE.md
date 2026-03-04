# LLM-as-a-Judge for Search Evaluation

This document explains how to use LLM-as-a-Judge for generating relevance judgements in the semantic e-commerce ranking tutorial.

**References:**
- [Vespa Blog: Improving Retrieval with LLM-as-a-Judge](https://blog.vespa.ai/improving-retrieval-with-llm-as-a-judge/)
- [Evaluation Framework Documentation](QUERIES.md#evaluation-framework)

---

## Table of Contents

1. [What is LLM-as-a-Judge?](#what-is-llm-as-a-judge)
2. [Why Use LLM-as-a-Judge?](#why-use-llm-as-a-judge)
3. [How It Works in This Tutorial](#how-it-works-in-this-tutorial)
4. [Setting Up LLM-as-a-Judge](#setting-up-llm-as-a-judge)
5. [Using create_judgements.py](#using-create_judgementspy)
6. [Understanding Judgements](#understanding-judgements)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Alternative Evaluation Methods](#alternative-evaluation-methods)

---

## What is LLM-as-a-Judge?

**LLM-as-a-Judge** is an evaluation technique that uses Large Language Models (LLMs) to automatically assess the relevance of search results. Instead of manually labeling thousands of query-document pairs, an LLM judges whether each document is relevant to a query.

### Traditional Evaluation

**Manual Labelling:**
- Human annotators review each query-document pair
- Time-consuming and expensive
- Requires domain expertise
- Difficult to scale

**LLM-as-a-Judge:**
- LLM automatically judges relevance
- Fast and cost-effective
- Can handle large-scale evaluation
- Consistent and reproducible

### How It Works

1. **Query Execution**: Run queries against your search system
2. **Result Collection**: Gather top results for each query
3. **LLM Judgement**: Send query + documents to LLM for relevance scoring
4. **Judgement Storage**: Save judgements for evaluation metrics

**Example:**
```
Query: "blue t-shirt"
Results: [Product A, Product B, Product C]

LLM judges:
- Product A: 3 (excellent match)
- Product B: 2 (good match)
- Product C: 1 (possible match)
```

---

## Why Use LLM-as-a-Judge?

### Benefits

**✅ Scalability:**
- Evaluate thousands of queries quickly
- No need for large annotation teams
- Can re-evaluate as system changes

**✅ Cost-Effective:**
- Much cheaper than human annotation
- Pay per API call (OpenAI, etc.)
- No need to hire annotators

**✅ Consistency:**
- Same criteria applied to all judgements
- Reproducible results
- No inter-annotator disagreement

**✅ Flexibility:**
- Easy to adjust evaluation criteria
- Can evaluate different aspects (relevance, quality, etc.)
- Works across domains

### Limitations

**⚠️ Model Dependency:**
- Quality depends on LLM capabilities
- May not match human judgement perfectly
- Requires API access (costs, rate limits)

**⚠️ Bias:**
- LLM may have biases from training data
- May favor certain types of content
- Should validate against human judgements when possible

**⚠️ Context Window:**
- Limited by LLM context size
- May need to batch documents
- Longer documents may be truncated

---

## How It Works in This Tutorial

### Architecture

```
┌─────────────┐
│ queries.csv │  Test queries
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ create_judgements.py│
│                     │
│ 1. Load queries     │
│ 2. Query Vespa      │──┐
│ 3. Get results      │  │
│ 4. Batch documents  │  │
│ 5. Call OpenAI API  │◄─┘
│ 6. Parse ratings    │
│ 7. Save judgements  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────┐
│ judgements.csv  │  Relevance judgements
└─────────────────┘
       │
       ▼
┌─────────────┐
│ evaluate.py │  Compute metrics (NDCG, MRR, etc.)
└─────────────┘
```

### Implementation Details

**Script**: `evaluation/create_judgements.py`

**Key Features:**
- Supports multiple query functions (vector + lexical search)
- Batches documents for efficient API usage
- Deduplicates results across query functions
- Saves judgements incrementally (resume on failure)
- Uses OpenAI GPT models for judgement

**Rating Scale:**
- **3**: Excellent match - directly answers the query
- **2**: Good match - relevant and useful
- **1**: Possible match - could be relevant for some users
- **0**: Irrelevant - does not answer the query

---

## Setting Up LLM-as-a-Judge

### Prerequisites

1. **OpenAI API Key:**
   - Sign up at https://platform.openai.com
   - Get API key from https://platform.openai.com/api-keys
   - Ensure you have credits/quota

2. **Python Dependencies:**
   ```bash
   cd evaluation
   pip install -r requirements.txt
   ```

   **Required packages:**
   - `openai` - OpenAI API client
   - `requests` - HTTP requests to Vespa
   - `python-dotenv` - Environment variable management
   - `csv` - CSV file handling

3. **Vespa Configuration:**
   - Vespa application deployed and accessible
   - Data fed to Vespa
   - Queries return results

### Configuration

**Create `.env` file** (copy from `env.example`):

```bash
# Vespa Configuration
VESPA_ENDPOINT=https://your-endpoint.vespa-app.cloud
VESPA_CERT_PATH=/path/to/cert.pem
VESPA_KEY_PATH=/path/to/key.pem

# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key-here

# Optional
HITS=20  # Documents to evaluate per query
QUERIES_FILE=queries.csv
JUDGEMENTS_FILE=judgements.csv
```

**Environment Variables:**
```bash
export OPENAI_API_KEY=sk-your-api-key-here
export VESPA_ENDPOINT=https://your-endpoint.vespa-app.cloud
export VESPA_CERT_PATH=/path/to/cert.pem
export VESPA_KEY_PATH=/path/to/key.pem
```

---

## Using create_judgements.py

### Basic Usage

**Step 1: Prepare Queries**

Create `queries.csv`:
```csv
query_id,query_text
q1,blue t-shirt
q2,comfortable running shoes
q3,home decor items
```

**Step 2: Run Script**

```bash
cd evaluation
python create_judgements.py
```

**What happens:**
1. Loads queries from `queries.csv`
2. For each query:
   - Executes query against Vespa (vector + lexical search)
   - Gets top results
   - Batches documents (10 per batch)
   - Sends to OpenAI for judgement
   - Saves judgements to `judgements.csv`
3. Shows progress and statistics

**Output:**
```
Loading queries...
Found 0 existing judgements
Processing query 1/3: blue t-shirt
  vector_search: 20 documents (20 new, 0 duplicates)
  lexical_search: 20 documents (15 new, 5 duplicates)
  Combined: 35 unique documents (from 40 total)
Evaluating batch 1/4 (10 docs)...
Evaluating batch 2/4 (10 docs)...
...
Saving 35 new judgements for query q1...
```

### Configuration Options

**Query Functions:**

Edit `create_judgements.py`:
```python
# Single query function
QUERY_FUNCTIONS = [evaluate.vector_search]

# Multiple query functions (combines results)
QUERY_FUNCTIONS = [evaluate.vector_search, evaluate.lexical_search]
```

**Batch Size:**

Controls how many documents are judged per API call:
```python
batch_size = 10  # Default: 10 documents per batch
```

**Hits per Query:**

Number of documents to retrieve and judge:
```bash
export HITS=50  # Default: 100
```

### Resuming Failed Runs

The script automatically resumes:
- Checks existing judgements in `judgements.csv`
- Skips already-judged query-document pairs
- Continues from where it left off

**To restart completely:**
```bash
rm judgements.csv
python create_judgements.py
```

---

## Understanding Judgements

### Judgement Format

**CSV Structure** (`judgements.csv`):
```csv
query_id,document_id,rating
q1,product123,3
q1,product456,2
q1,product789,1
q2,product321,3
```

**Fields:**
- **`query_id`**: Query identifier from `queries.csv`
- **`document_id`**: Product ID (ProductID field)
- **`rating`**: Relevance score (0-3)

### Rating Interpretation

**3 - Excellent Match:**
- Directly answers the query
- Highly relevant
- User would be satisfied

**Example:**
```
Query: "blue t-shirt"
Product: "Blue Cotton T-Shirt" → Rating: 3
```

**2 - Good Match:**
- Relevant and useful
- Meets query intent
- User would find it helpful

**Example:**
```
Query: "blue t-shirt"
Product: "Navy Blue Polo Shirt" → Rating: 2
```

**1 - Possible Match:**
- Could be relevant for some users
- Partial match
- May not fully satisfy query

**Example:**
```
Query: "blue t-shirt"
Product: "Blue Jeans" → Rating: 1
```

**0 - Irrelevant:**
- Does not answer the query
- Not relevant
- User would not find it useful

**Example:**
```
Query: "blue t-shirt"
Product: "Red Dress" → Rating: 0
```

### Using Judgements

**For Evaluation:**
```bash
python evaluate.py
```

**Metrics computed:**
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Recall@10**: Recall at 10 results

**For Analysis:**
- Compare ratings across queries
- Identify patterns (which queries are hard?)
- Validate search quality

---

## Best Practices

### 1. Query Selection

**Choose diverse queries:**
- Different query types (specific, broad, natural language)
- Various difficulty levels
- Representative of real user queries

**Avoid:**
- Too many similar queries
- Queries with no relevant results
- Queries that are too vague

### 2. Document Selection

**Retrieve enough candidates:**
- Set `HITS` high enough (20-100)
- Use multiple query functions for better coverage
- Include both vector and lexical results

**Balance:**
- Too few: May miss relevant documents
- Too many: Expensive API calls, slower

### 3. Batch Size

**Optimal batch size:**
- **10-20 documents**: Good balance
- **Too small**: Many API calls, slower
- **Too large**: May hit context limits, less efficient

### 4. Prompt Quality

**Current prompt** (in `create_judgements.py`):
```python
prompt = f"""
Please rate the relevance of each of the following products to the search query on a scale of 0-3.

Query: "{query_text}"

Products (JSON array):
{products_json}

Rating scale:
- 3: Excellent match - directly answers the query
- 2: Good match - relevant and useful
- 1: Possible match - could be relevant for some users
- 0: Irrelevant - does not answer the query

Respond with exactly a JSON array of objects. Each object must have:
- "ProductID": string
- "rating": number (0, 1, 2, or 3)
"""
```

**Improvements:**
- Add examples (few-shot learning)
- Specify domain-specific criteria
- Include product context (price, brand, etc.)

### 5. Cost Management

**OpenAI API Costs:**
- Pay per token (input + output)
- Batch documents to reduce calls
- Use cheaper models when possible
- Monitor usage via OpenAI dashboard

**Estimate:**
- ~100 queries × 20 documents = 2000 judgements
- ~$5-20 depending on model and document length

### 6. Validation

**Validate against human judgements:**
- Sample 10-20 judgements for human review
- Compare LLM vs. human ratings
- Adjust if significant disagreement

**Check for bias:**
- Review ratings for different query types
- Ensure fair evaluation across product categories
- Watch for systematic errors

---

## Troubleshooting

### OpenAI API Errors

**Error**: `Invalid API key`

**Solutions:**
1. Verify `OPENAI_API_KEY` is set correctly
2. Check API key is active (not expired)
3. Ensure key has sufficient credits
4. Verify key format: `sk-...`

**Error**: `Rate limit exceeded`

**Solutions:**
1. Reduce batch size
2. Add delays between API calls
3. Use rate limiting library
4. Upgrade OpenAI plan for higher limits

**Error**: `Context length exceeded`

**Solutions:**
1. Reduce batch size (fewer documents per call)
2. Truncate long product descriptions
3. Use shorter product fields
4. Consider using models with larger context windows

### Vespa Connection Issues

**Error**: Connection timeout or certificate errors

**Solutions:**
1. Verify `VESPA_ENDPOINT` is correct
2. Check certificate paths are valid
3. Ensure Vespa application is accessible
4. Test connection manually: `curl -k https://endpoint/search/`

### Judgement Quality Issues

**Issue**: Ratings seem inconsistent or wrong

**Solutions:**
1. Review prompt quality
2. Add examples to prompt (few-shot)
3. Try different LLM models
4. Validate sample judgements manually
5. Check if documents have sufficient information

**Issue**: All ratings are 0 or 1

**Solutions:**
1. Check if queries are too specific
2. Verify documents are relevant to queries
3. Review prompt - may be too strict
4. Check if product data is complete

### Script Errors

**Error**: `FileNotFoundError: queries.csv`

**Solutions:**
1. Ensure `queries.csv` exists in `evaluation/` directory
2. Check file path in configuration
3. Verify CSV format is correct

**Error**: JSON parsing errors

**Solutions:**
1. Check OpenAI response format
2. Add error handling for malformed JSON
3. Review prompt to ensure JSON output
4. Try different models (some are better at JSON)

---

## Alternative Evaluation Methods

### Human Annotation

**When to use:**
- Need highest quality judgements
- Small-scale evaluation (< 100 queries)
- Domain-specific expertise required
- Validating LLM judgements

**Pros:**
- Highest quality
- Domain expertise
- Understands context

**Cons:**
- Expensive
- Time-consuming
- Hard to scale
- Inter-annotator disagreement

### Click-Through Data

**When to use:**
- Have user interaction data
- Large-scale evaluation
- Real user behavior

**Pros:**
- Real user signals
- Large scale
- Automatic collection

**Cons:**
- Requires user traffic
- May have bias (position, presentation)
- Not always available

### Synthetic Judgements

**When to use:**
- Testing specific scenarios
- Controlled experiments
- No access to real data

**Pros:**
- Fast to generate
- Controlled conditions
- Reproducible

**Cons:**
- May not reflect real usage
- Limited scenarios
- Requires domain knowledge

### Hybrid Approach

**Best practice:**
- Use LLM-as-a-Judge for large-scale evaluation
- Validate with human annotation (sample)
- Combine with click-through data when available
- Use synthetic judgements for specific tests

---

## Additional Resources

- [Vespa Blog: Improving Retrieval with LLM-as-a-Judge](https://blog.vespa.ai/improving-retrieval-with-llm-as-a-judge/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Evaluation Framework Guide](QUERIES.md#evaluation-framework)
- [Vespa Evaluation Framework](https://vespa-engine.github.io/pyvespa/evaluating-vespa-application-cloud.html)

**Related Documentation:**
- [`docs/QUERIES.md`](QUERIES.md) – Query patterns and evaluation
- [`docs/RANKING.md`](RANKING.md) – Ranking concepts
- [`evaluation/create_judgements.py`](../evaluation/create_judgements.py) – Implementation
