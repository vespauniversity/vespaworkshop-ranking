# Embeddings and Models

This note complements the semantic ranking docs with model choices, sizing, and practical tips for embeddings in Vespa.

**References:**
- [Vespa Embedding Overview](https://docs.vespa.ai/en/embedding.html)
- [Hugging Face Embedder](https://docs.vespa.ai/en/embedding.html#huggingface-embedder)
- [Distance Metrics](https://docs.vespa.ai/en/reference/schema-reference.html#distance-metric)

---

## Models Used in this Tutorial

**Snowflake Arctic Embed XS** (default)
- Dim: 384
- Size: ~50MB
- Normalized outputs (use `prenormalized-angular` / cosine)
- Multilingual, fast, good quality for demo and teaching

**Why Arctic XS here?**
- Small and quick to download
- Good balance of quality and speed for live labs
- Works well with cosine similarity

---

## Other Model Options (when to consider)

- **Arctic Embed Small/Medium**: Higher quality, slower, larger download.
- **Sentence Transformers (e.g., all-MiniLM-L6-v2, all-mpnet-base-v2)**: Strong English performance; choose `innerproduct` if not normalized, or set `normalize:true` and use `prenormalized-angular`.
- **OpenAI text-embedding-3-small / 3-large**: Hosted, high quality; use cloud embedder or precompute offline; dims 1536 / 3072.
- **Domain-tuned models**: If you have in-domain data (e.g., fashion, electronics), fine-tune and serve via ONNX embedder.

**Selection tips:**
- Latency budget tight → smaller models (XS/Small).
- Quality priority → larger models or domain-tuned.
- Multilingual needs → models advertised as multilingual.
- GPU vs CPU → ONNX + CPU is fine for XS/Small; larger models may need GPU or batching.

---

## Distance Metric Cheat Sheet

- Normalized outputs → `prenormalized-angular` (cosine via dot product).
- Unnormalized outputs → `innerproduct` or `euclidean` (check model docs).
- If you enable `<normalize>true</normalize>` in the embedder, prefer `prenormalized-angular`.

---

## Operational Notes

- First deploy downloads the model; expect a one-time delay.
- Re-feed documents after changing embedding fields or model.
- Keep embedding dimensions consistent between query and document fields.
- For ANN quality, set `targetHits` ~10x desired hits; tune per latency needs.

---

## When to Precompute vs. Inline Embed

- **Inline (schema `embed arctic`)**: Easiest, consistent, handled by Vespa at index time.
- **Precompute offline**: Use when you need custom pipelines, expensive models, or batching; write tensors directly to fields.

---

## Quick Model Comparison (rule of thumb)

| Model | Dim | Size | Speed | Notes |
|-------|-----|------|-------|-------|
| Arctic XS | 384 | ~50MB | Fast | Default in tutorial, normalized |
| all-MiniLM-L6-v2 | 384 | ~80MB | Fast | Strong English; normalize for cosine |
| all-mpnet-base-v2 | 768 | ~420MB | Medium | Higher quality, slower |
| text-embedding-3-small | 1536 | Hosted | Depends | High quality; API cost/latency |
| text-embedding-3-large | 3072 | Hosted | Slower | Highest quality; expensive |

---

## How to swap models (Hugging Face embedder)

1. Update `services.xml` component URLs to the new model/tokenizer.
2. Adjust embedding field dimensions in `product.sd` to match the new model.
3. Redeploy and re-feed data.
4. Verify `distance-metric` matches normalization.

Example change (MiniLM, 384 dim):
```xml
<component id="mini" type="hugging-face-embedder">
    <transformer-model url="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.onnx"/>
    <tokenizer-model url="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/raw/main/tokenizer.json"/>
    <normalize>true</normalize>
    <pooling-strategy>cls</pooling-strategy>
</component>
```
And in schema:
```vespa
field ProductName_embedding type tensor<float>(x[384]) {
    indexing: input ProductName | embed mini | attribute | index
    attribute { distance-metric: prenormalized-angular }
}
```
