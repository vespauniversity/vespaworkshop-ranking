# LightGBM Reranking Guide

This document provides a complete guide to training and deploying LightGBM rerankers for Chapter 3: Hybrid Search with Learned Reranking.

## Table of Contents

- [Overview](#overview)
- [Reranking Pipeline](#reranking-pipeline)
- [Obtaining Relevance Judgements](#obtaining-relevance-judgements)
- [Collecting Training Data](#collecting-training-data)
- [Training LightGBM](#training-lightgbm)
- [Deploying the Model](#deploying-the-model)
- [Evaluating Rerankers](#evaluating-rerankers)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Production Best Practices](#production-best-practices)

---

## Overview

### What is Learned Reranking?

**Learned reranking** uses machine learning to optimize document ranking based on historical relevance data:

**Traditional ranking** (manual):
```vespa
expression: feature1() * 2.0 + feature2() * 1.5 + feature3()
```
- Manual feature weights
- Linear combination
- Hard to tune
- Suboptimal

**Learned ranking** (automatic):
```vespa
expression: lightgbm("model.json")
```
- Learned feature weights
- Non-linear combinations
- Optimized for ranking metrics
- Better performance

### Why LightGBM?

**LightGBM** (Light Gradient Boosting Machine) is ideal for ranking tasks:

**Advantages:**
- **Fast training**: Efficient on large datasets
- **High quality**: State-of-the-art ranking performance
- **Ranking optimization**: Built-in NDCG@k objective
- **Feature handling**: Automatic categorical encoding, interaction discovery
- **Vespa integration**: Native support via JSON export

**Alternatives:**
- **XGBoost**: Similar to LightGBM, slightly slower
- **Neural rerankers**: Better quality, much slower (BERT, cross-encoders)
- **Linear models**: Faster, lower quality (logistic regression, SVM)

---

## Reranking Pipeline

### Complete Pipeline

```
1. Define Queries
   ↓
2. Obtain Relevance Judgements (LLM-as-a-judge or manual)
   ↓
3. Split Train/Test Sets
   ↓
4. Collect Rank Features (query Vespa, extract summary-features)
   ↓
5. Train LightGBM Model (cross-validation, feature importance)
   ↓
6. Export Model to JSON
   ↓
7. Deploy to Vespa (add to app/models/)
   ↓
8. Create Rerank Profile (second-phase with lightgbm())
   ↓
9. Evaluate on Test Set (NDCG, MRR, etc.)
   ↓
10. A/B Test in Production
```

### Timeline

**Typical timeline for first reranker:**
- Day 1: Collect/create relevance judgements (4-8 hours)
- Day 2: Collect rank features and train model (2-4 hours)
- Day 3: Deploy, evaluate, and iterate (2-4 hours)

**Subsequent iterations:**
- Collect new training data: 1-2 hours
- Retrain model: 15-30 minutes
- Deploy and evaluate: 30 minutes

---

## Obtaining Relevance Judgements

### Relevance Judgement Format

**CSV format:**
```csv
query_id,query_text,document_id,relevance
1,comfortable running shoes,prod_123,3
1,comfortable running shoes,prod_456,2
1,comfortable running shoes,prod_789,0
2,blue jeans for women,prod_234,3
2,blue jeans for women,prod_567,1
```

**Columns:**
- `query_id`: Unique query identifier (integer)
- `query_text`: Query string
- `document_id`: Document identifier (matches Vespa document ID)
- `relevance`: Relevance grade (0-3)

**Relevance scale:**
- **3**: Highly relevant (perfect match)
- **2**: Relevant (good match)
- **1**: Marginally relevant (somewhat related)
- **0**: Not relevant (unrelated)

### Option 1: Manual Judgements

**Process:**
1. Define representative queries (20-100 queries)
2. Run queries against Vespa
3. Review top 20-50 results per query
4. Manually label each result (0-3 scale)
5. Save to CSV file

**Tools:**
- Spreadsheet (Google Sheets, Excel)
- Custom UI for labeling
- Annotation platforms (Label Studio, Prodigy)

**Effort:**
- 20 queries × 20 docs/query × 30 sec/doc = ~3 hours
- 100 queries × 20 docs/query × 30 sec/doc = ~17 hours

### Option 2: LLM-as-a-Judge

**Use LLM (GPT-4, Claude) to generate relevance judgements:**

**Advantages:**
- Much faster than manual labeling
- Consistent criteria
- Scalable to 1000s of queries

**Disadvantages:**
- May not match human judgements perfectly
- Requires API access and cost
- Need to validate quality

**Process:**
See Chapter 2 (`semantic_ecommerce_ranking_app/evaluation/create_judgements.py`) for LLM-as-a-judge implementation.

**Example prompt:**
```
You are evaluating search result relevance.

Query: "comfortable running shoes"
Product: "Nike Air Zoom Pegasus 38 - Men's running shoe with React foam cushioning"

Rate relevance on scale:
3 - Highly relevant (perfect match)
2 - Relevant (good match)
1 - Marginally relevant (somewhat related)
0 - Not relevant (unrelated)

Rating: 3
Explanation: Perfect match - specifically a comfortable running shoe.
```

### Option 3: Implicit Feedback

**Use user behavior as relevance signal:**

**Signals:**
- Click-through rate (CTR): Clicked = relevant
- Dwell time: Long view = relevant
- Add to cart / Purchase: High relevance
- Skip / immediate back: Not relevant

**Conversion:**
- No interaction: 0
- Click only: 1
- Add to cart: 2
- Purchase: 3

**Advantages:**
- Reflects actual user preferences
- No manual labeling
- Large scale

**Disadvantages:**
- Requires production traffic
- Position bias (top results get more clicks)
- Sparse (most docs not interacted with)

**Recommended approach:**
Start with manual/LLM judgements → Train initial model → Collect implicit feedback → Retrain with real user data

---

## Collecting Training Data

### Environment Setup

**Install dependencies:**
```bash
cd train_reranker
pip install -r requirements.txt
```

**Set environment variables:**
```bash
# Copy template
cp env.example .env

# Edit .env
export VESPA_ENDPOINT=https://your-endpoint.vespa-app.cloud
export VESPA_CERT_PATH=/path/to/cert.pem
export VESPA_KEY_PATH=/path/to/key.pem
```

**Or use helper script:**
```bash
source prepare_env.sh
```

### Split Judgements

**Purpose:** Prevent data leakage and measure generalization

**Script:** `split_judgements.py`

```bash
python split_judgements.py \
  --input_file /path/to/judgements.csv \
  --train_output judgements_train.csv \
  --test_output judgements_test.csv \
  --test_ratio 0.2 \
  --random_seed 42
```

**Parameters:**
- `--input_file`: Full judgement file
- `--train_output`: Training set output (80%)
- `--test_output`: Test set output (20%)
- `--test_ratio`: Fraction for test set (0.2 = 20%)
- `--random_seed`: Random seed for reproducibility

**How it works:**
- Splits by query (all docs for a query in same set)
- Stratified splitting (balanced relevance distribution)
- No overlap between train and test

**Verify split:**
```bash
wc -l judgements_train.csv judgements_test.csv
# Train: ~400 pairs (80%)
# Test: ~100 pairs (20%)

# Check no query overlap
awk -F',' '{print $1}' judgements_train.csv | sort -u > train_queries.txt
awk -F',' '{print $1}' judgements_test.csv | sort -u > test_queries.txt
comm -12 train_queries.txt test_queries.txt
# Should output nothing (no overlap)
```

### Extract Rank Features

**Purpose:** Query Vespa to get rank features for training

**Script:** `create_prediction_data.py`

```bash
python create_prediction_data.py \
  --judgements_file judgements_train.csv \
  --output_file training_data.csv \
  --ranking_profile hybrid \
  --hits 100
```

**Parameters:**
- `--judgements_file`: Input judgements (train or test)
- `--output_file`: Output CSV with features
- `--ranking_profile`: Rank profile to use (must have summary-features)
- `--hits`: Number of results to retrieve per query

**How it works:**
1. For each query in judgements:
   - Query Vespa with specified rank profile
   - Retrieve top N results
   - Extract `summaryfeatures` from response
2. Join with relevance labels from judgements
3. Create feature matrix with labels
4. Save to CSV

**Output format:**
```csv
query_id,document_id,relevance,native_rank_name,native_rank_description,closeness_productname,closeness_description,AverageRating,Price
1,prod_123,3,2.345,1.876,0.912,0.784,4.5,2999
1,prod_456,2,1.678,1.234,0.723,0.654,4.2,1999
1,prod_789,0,0.456,0.312,0.234,0.189,3.8,3499
```

**Important:**
- Rank profile must expose features via `summary-features`
- Missing features will have null/0 values
- Check output for completeness before training

**Troubleshooting:**

**Empty features:**
```bash
# Verify rank profile has summary-features
vespa query 'yql=select * from product where true' 'ranking.profile=hybrid' 'hits=1'

# Check response includes summaryfeatures
```

**Connection errors:**
```bash
# Verify Vespa endpoint
curl -k --cert $VESPA_CERT_PATH --key $VESPA_KEY_PATH $VESPA_ENDPOINT/search/

# Check certificates
ls -lh $VESPA_CERT_PATH $VESPA_KEY_PATH
```

---

## Training LightGBM

### Training Script

**Script:** `train_lightgbm.py`

```bash
python train_lightgbm.py \
  --input_file training_data.csv \
  --output_model ../app/models/lightgbm_model.json \
  --target_col relevance \
  --drop_cols query_id,document_id \
  --folds 5 \
  --learning_rate 0.05 \
  --num_leaves 31 \
  --max_depth -1 \
  --max_rounds 1000 \
  --early_stop 50 \
  --seed 42
```

### Key Parameters

**Data parameters:**
- `--input_file`: Training data CSV (from create_prediction_data.py)
- `--target_col`: Target column name (default: "relevance")
- `--drop_cols`: Columns to exclude from features (default: "query_id,document_id")

**Cross-validation:**
- `--folds`: Number of CV folds (default: 5)
- `--seed`: Random seed for reproducibility

**Model hyperparameters:**
- `--learning_rate`: Learning rate (default: 0.05)
  - Lower: Slower training, less overfitting
  - Higher: Faster training, may overfit
  - Range: 0.01 - 0.1

- `--num_leaves`: Max leaves per tree (default: 31)
  - Lower: Simpler model, less overfitting
  - Higher: More complex model, better fit
  - Range: 15 - 63

- `--max_depth`: Max tree depth (default: -1 = unlimited)
  - Controls tree complexity
  - Use with num_leaves for regularization
  - Range: 3 - 10 (or -1)

- `--max_rounds`: Max boosting iterations (default: 1000)
  - Early stopping will stop earlier if no improvement

- `--early_stop`: Early stopping rounds (default: 50)
  - Stop if no improvement for N rounds
  - Prevents overfitting

**Output:**
- `--output_model`: Model output path (JSON format for Vespa)
- `--output_importance`: Feature importance CSV (default: auto-generated)

### Training Output

**Console output:**
```
Loaded 450 rows × 9 columns
Training with 5-fold stratified cross-validation

Fold 1/5:
Training until validation scores don't improve for 50 rounds
[50]  valid's ndcg@10: 0.7234
[100] valid's ndcg@10: 0.7456
[150] valid's ndcg@10: 0.7523
[200] valid's ndcg@10: 0.7545
Early stopping at round 215
Fold 1 metrics:
  NDCG@10: 0.7545
  MSE: 0.4234
  MAE: 0.5123

Fold 2/5:
...

Cross-validation results:
  Mean NDCG@10: 0.7589 (±0.0123)
  Mean MSE: 0.4156 (±0.0234)
  Mean MAE: 0.5078 (±0.0145)

Training final model on full dataset...
Final model exported to: ../app/models/lightgbm_model.json

Feature importance (top 10):
  closeness_productname: 0.3245
  native_rank_name: 0.2156
  closeness_description: 0.1876
  native_rank_description: 0.1234
  AverageRating: 0.0987
  Price: 0.0502
```

### Interpreting Results

**Metrics:**

**NDCG@10** (primary metric):
- **Range:** 0.0 - 1.0
- **Interpretation:**
  - < 0.5: Poor ranking
  - 0.5 - 0.7: Reasonable
  - 0.7 - 0.8: Good
  - > 0.8: Excellent
- **Goal:** Maximize on validation set

**MSE** (Mean Squared Error):
- **Range:** 0.0 - ∞
- **Interpretation:** Lower is better
- **Use:** Regression error on relevance scores

**MAE** (Mean Absolute Error):
- **Range:** 0.0 - ∞
- **Interpretation:** Lower is better
- **Use:** Average prediction error

**Cross-validation variance:**
- **Low variance** (±0.01): Stable model, good generalization
- **High variance** (±0.1+): Unstable, may be overfitting

**Feature importance:**
- Shows which features model relies on most
- High importance: Critical for ranking
- Low importance: Consider removing (reduces complexity)

### Feature Importance Analysis

**Example output:**
```
Feature importance (gain):
  closeness_productname: 0.32  ← Semantic match on name
  native_rank_name: 0.22        ← Lexical match on name
  closeness_description: 0.19   ← Semantic match on description
  native_rank_description: 0.15 ← Lexical match on description
  AverageRating: 0.08           ← Product quality
  Price: 0.04                   ← Price factor
```

**Insights:**
1. **Semantic features dominate** (0.32 + 0.19 = 0.51 combined)
2. **ProductName more important** than Description (0.32 vs 0.19)
3. **Business signals** (rating, price) contribute but less critical

**Actions:**
- Keep high-importance features
- Consider removing very low importance features (< 0.02)
- Add more features in underperforming areas

---

## Deploying the Model

### Model Export

**LightGBM JSON format:**
The training script exports model to Vespa-compatible JSON:

```json
{
  "name": "lightgbm_model",
  "num_tree_per_iteration": 1,
  "max_feature_idx": 6,
  "objective": "regression",
  "average_output": false,
  "tree_info": [
    {
      "tree_index": 0,
      "num_leaves": 15,
      "split_feature": [0, 1, 2, ...],
      "threshold": [0.5, 0.3, 0.8, ...],
      "decision_type": [2, 2, 2, ...],
      "left_child": [1, 3, 5, ...],
      "right_child": [2, 4, 6, ...],
      "leaf_value": [0.12, -0.05, 0.34, ...],
      ...
    },
    ...
  ]
}
```

### Add Model to Application

**1. Create models directory:**
```bash
mkdir -p app/models
```

**2. Copy trained model:**
```bash
cp train_reranker/output/lightgbm_model.json app/models/
```

**3. Verify model file:**
```bash
ls -lh app/models/lightgbm_model.json
# Should be 100KB - 10MB depending on model size

# Check valid JSON
python -m json.tool app/models/lightgbm_model.json > /dev/null
echo $?  # Should output 0 (success)
```

### Create Rerank Profile

**File:** `app/schemas/product/rerank.profile`

```vespa
rank-profile rerank inherits hybrid {
    ## Second-phase reranking with LightGBM
    second-phase {
        rerank-count: 20
        expression: lightgbm("lightgbm_model.json")
    }
}
```

**Key points:**
- `inherits hybrid`: Reuses first-phase and all functions
- `rerank-count: 20`: Rerank top 20 from first phase
- `lightgbm("lightgbm_model.json")`: Model filename (relative to app/models/)

**Alternative (without inheritance):**
```vespa
rank-profile rerank {
    ## Define all functions (copy from hybrid.profile)
    function native_rank_name() {
        expression: nativeRank(ProductName)
    }
    # ... other functions ...

    ## First-phase
    first-phase {
        expression: native_rank_name() + closeness_productname() + ...
    }

    ## Second-phase
    second-phase {
        rerank-count: 20
        expression: lightgbm("lightgbm_model.json")
    }
}
```

### Deploy to Vespa

```bash
cd app

# Deploy application
vespa deploy --wait 900

# Verify deployment
vespa status

# Check model loaded
vespa query \
  'yql=select * from product where true' \
  'ranking.profile=rerank' \
  'hits=5'
```

**Deployment checklist:**
- [ ] Model file exists in `app/models/lightgbm_model.json`
- [ ] Rerank profile created
- [ ] Profile inherits from hybrid (or defines all features)
- [ ] Model filename matches profile reference
- [ ] Deployment successful (`vespa status` shows green)
- [ ] Test query returns results

---

## Evaluating Rerankers

### Evaluation Script

**Script:** `evaluate_model.py`

```bash
python evaluate_model.py \
  --judgements_file judgements_test.csv \
  --ranking_profile rerank \
  --baseline_profile hybrid \
  --output_file evaluation_results.csv
```

**Parameters:**
- `--judgements_file`: Test set judgements (held-out data)
- `--ranking_profile`: Profile to evaluate (e.g., "rerank")
- `--baseline_profile`: Baseline for comparison (e.g., "hybrid")
- `--output_file`: Results CSV (optional)

### Evaluation Metrics

**NDCG@10** (primary metric):
```python
def ndcg_at_k(relevances, k=10):
    """
    relevances: List of relevance scores in ranked order
    k: Cutoff for evaluation
    """
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted(relevances, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0
```

**MRR** (Mean Reciprocal Rank):
```python
def mrr(relevances):
    """First relevant result position"""
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0
```

**Recall@10**:
```python
def recall_at_k(retrieved_relevant, total_relevant, k=10):
    """Fraction of relevant docs in top k"""
    return len(retrieved_relevant[:k]) / total_relevant if total_relevant > 0 else 0.0
```

### Example Output

```
Evaluating ranking profile: rerank
Querying Vespa for 50 test queries...

Reranker results:
  NDCG@10: 0.7845
  MRR: 0.6234
  Recall@10: 0.8123
  Precision@10: 0.6500
  MAP: 0.7123

Baseline (hybrid) results:
  NDCG@10: 0.7123
  MRR: 0.5845
  Recall@10: 0.7856
  Precision@10: 0.6200
  MAP: 0.6789

Improvement:
  NDCG@10: +10.1% ✓
  MRR: +6.7% ✓
  Recall@10: +3.4% ✓
  Precision@10: +4.8% ✓
  MAP: +4.9% ✓

Statistical significance: p < 0.01 (significant)
```

### Interpreting Results

**Good improvements:**
- NDCG@10: +5% to +15%
- MRR: +3% to +10%
- Consistent improvements across metrics

**Warning signs:**
- Train NDCG much higher than test (overfitting)
- Test NDCG below baseline (model not learning)
- High variance across test queries (unstable)

**Actions:**
- **Good results**: Deploy to production, A/B test
- **Overfitting**: Regularize (lower learning rate, fewer trees)
- **Underfitting**: More features, more training data
- **Unstable**: More diverse training data

---

## Hyperparameter Tuning

### Key Hyperparameters

**Learning rate** (`--learning_rate`):
- **Default**: 0.05
- **Range**: 0.01 - 0.1
- **Effect**: Lower = slower training, less overfitting
- **Tuning**: Start at 0.05, decrease if overfitting

**Number of leaves** (`--num_leaves`):
- **Default**: 31
- **Range**: 15 - 63
- **Effect**: Higher = more complex model
- **Tuning**: Increase if underfitting, decrease if overfitting

**Max depth** (`--max_depth`):
- **Default**: -1 (unlimited)
- **Range**: 3 - 10
- **Effect**: Limits tree depth
- **Tuning**: Set to 5-7 if overfitting

**Early stopping** (`--early_stop`):
- **Default**: 50
- **Range**: 20 - 100
- **Effect**: Stop training when no improvement
- **Tuning**: Lower for faster training, higher for better convergence

### Grid Search

**Manual grid search:**
```bash
# Define hyperparameter grid
learning_rates=(0.01 0.05 0.1)
num_leaves=(15 31 63)

# Test all combinations
for lr in "${learning_rates[@]}"; do
  for leaves in "${num_leaves[@]}"; do
    echo "Testing lr=$lr, leaves=$leaves"
    python train_lightgbm.py \
      --input_file training_data.csv \
      --output_model models/model_lr${lr}_leaves${leaves}.json \
      --learning_rate $lr \
      --num_leaves $leaves \
      --folds 5 | tee logs/lr${lr}_leaves${leaves}.log
  done
done

# Compare results
grep "Mean NDCG@10" logs/*.log | sort -k3 -rn
```

**Automated hyperparameter tuning (Optuna):**
```python
import optuna
import lightgbm as lgb

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
    }

    # Train model with params
    # Return validation NDCG@10
    return ndcg_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best params: {study.best_params}")
print(f"Best NDCG@10: {study.best_value}")
```

### Recommendations

**Start with defaults:**
```bash
python train_lightgbm.py \
  --input_file training_data.csv \
  --learning_rate 0.05 \
  --num_leaves 31 \
  --max_depth -1
```

**If overfitting** (train >> test):
```bash
python train_lightgbm.py \
  --learning_rate 0.01 \
  --num_leaves 15 \
  --max_depth 5
```

**If underfitting** (both train and test low):
```bash
python train_lightgbm.py \
  --learning_rate 0.1 \
  --num_leaves 63 \
  --max_depth -1
```

---

## Production Best Practices

### Model Lifecycle

**1. Initial deployment:**
- Train on diverse, high-quality judgements
- Evaluate on held-out test set
- A/B test against baseline
- Monitor online metrics (CTR, conversion)

**2. Monitoring:**
- Track NDCG on test set over time
- Monitor feature distributions (detect drift)
- Log predictions for analysis
- Alert on performance degradation

**3. Retraining:**
- Quarterly retraining recommended
- Trigger on significant drift or new features
- Always evaluate on fresh test set
- A/B test new model before full deployment

### Data Quality

**Essential:**
- Diverse queries (cover all query types)
- Accurate labels (validate subset manually)
- Balanced relevance distribution (not all 0 or 3)
- Representative of production (real user queries)

**Monitoring:**
```python
# Check relevance distribution
import pandas as pd
df = pd.read_csv('judgements.csv')
print(df['relevance'].value_counts(normalize=True))

# Should see reasonable distribution
# 0: ~40-60%
# 1: ~10-20%
# 2: ~15-25%
# 3: ~10-20%
```

### A/B Testing

**Setup:**
```vespa
# Control group: baseline hybrid
ranking.profile=hybrid

# Treatment group: LightGBM reranker
ranking.profile=rerank
```

**Metrics to track:**
- Click-through rate (CTR)
- Conversion rate
- Revenue per search
- User engagement (time on site)
- Query reformulation rate

**Minimum sample size:**
- 1000+ queries per group
- 1-2 weeks duration
- Statistical significance: p < 0.05

### Model Versioning

**Naming convention:**
```
lightgbm_model_v1_20250101.json
lightgbm_model_v2_20250215.json
```

**Rollback strategy:**
```vespa
rank-profile rerank_v1 inherits hybrid {
    second-phase {
        expression: lightgbm("lightgbm_model_v1_20250101.json")
    }
}

rank-profile rerank_v2 inherits hybrid {
    second-phase {
        expression: lightgbm("lightgbm_model_v2_20250215.json")
    }
}
```

---

## Summary

**Key takeaways:**
1. **Relevance judgements** are critical - quality > quantity
2. **Train/test split** prevents overfitting and measures generalization
3. **LightGBM** provides state-of-the-art ranking with minimal tuning
4. **Cross-validation** ensures robust performance estimates
5. **Feature importance** guides feature engineering
6. **A/B testing** validates offline improvements translate to online gains

**Common pitfalls:**
- Insufficient training data (< 100 query-doc pairs)
- Data leakage (same query in train and test)
- Ignoring feature engineering (using raw features only)
- Not monitoring in production (model drift)
- Skipping A/B test (offline metrics don't guarantee online success)

**Resources:**
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Vespa LightGBM Integration](https://docs.vespa.ai/en/lightgbm.html)
- [Learning to Rank Tutorial](https://docs.vespa.ai/en/learning-to-rank.html)
- [Ranking Evaluation Metrics](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
