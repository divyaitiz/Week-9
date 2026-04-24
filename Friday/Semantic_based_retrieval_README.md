# Semantic-Based Retrieval with Embeddings & FAISS

A hands-on codelab exploring semantic similarity and document retrieval using two embedding strategies — **sentence embeddings** and **word embeddings** — across two model families: **SentenceTransformer** and **BERT**.

---

## Table of Contents

1. [Overview](#overview)
2. [Setup & Dependencies](#setup--dependencies)
3. [Models Used](#models-used)
4. [Notebook Structure](#notebook-structure)
5. [Results Comparison](#results-comparison)
   - [Sentence Transformer — Sentence Embeddings](#1-sentence-transformer--sentence-embeddings)
   - [Sentence Transformer — Word Embeddings](#2-sentence-transformer--word-embeddings)
   - [BERT — Sentence Embeddings](#3-bert--sentence-embeddings)
   - [BERT — Word Embeddings](#4-bert--word-embeddings)
6. [Key Takeaways](#key-takeaways)

---

## Overview

This notebook demonstrates **semantic retrieval** — retrieving documents based on meaning rather than exact keyword overlap. It uses FAISS (Facebook AI Similarity Search) as a vector store and compares how different embedding methods affect retrieval quality, especially in ambiguous cases like the polysemous word **"bank"** (river bank vs. financial bank).

---

## Setup & Dependencies

```bash
pip install sentence-transformers faiss-cpu transformers torch
```

| Library | Purpose |
|---|---|
| `sentence-transformers` | High-quality sentence-level embeddings |
| `faiss-cpu` | Efficient vector similarity search |
| `transformers` | BERT tokenizer and model |
| `torch` | Tensor operations and inference |

---

## Models Used

| Model | Source | Embedding Dimension |
|---|---|---|
| `all-MiniLM-L6-v2` | SentenceTransformers | **384** |
| `bert-base-cased` | Hugging Face Transformers | **768** |

---

## Notebook Structure

```
Part 1 — SentenceTransformer Sentence Embeddings
  └── Cosine similarity on 4 sentences
  └── FAISS retrieval on NLP document corpus
  └── Class activity: "bank" polysemy (FAISS search)

Part 2 — BERT Sentence Embeddings
  └── Tokenization with padding & attention masks
  └── [CLS] token as sentence representation
  └── Cosine similarity matrix on document_activity
  └── FAISS retrieval with BERT sentence embeddings

Part 3 — BERT Word Embeddings
  └── Per-token embeddings from last hidden state
  └── Contextual "bank" embeddings across sentences
  └── Pairwise similarity of all "bank" occurrences
  └── FAISS search using contextual word vectors

Part 4 — SentenceTransformer Word Embeddings
  └── Access to base transformer layer
  └── Token-level embeddings from the underlying model
  └── FAISS retrieval using word-level "bank" vectors
```

---

## Results Comparison

### Core Dataset — `document_activity`

```
1. "The bank is near the river."
2. "The cat is near the bank."
3. "Deposit 10000 RS in the bank."
4. "Deposit 10000 RS in the bank near river."
5. "Deposit 10000 RS in the bank near river bank."
```

Query: `"What does bank mean?"`

---

### 1. Sentence Transformer — Sentence Embeddings

**Model:** `all-MiniLM-L6-v2` | **Embedding dim:** 384 | **Method:** Mean-pooled token embeddings

**How it works:** The entire sentence is encoded into a single fixed-size vector. Retrieval is done by comparing the query vector to all document vectors in the FAISS index using L2 distance.

**Cosine Similarity Sample (4-sentence set):**

| Sentence A | Sentence B | Score |
|---|---|---|
| "The cat sits outside" | "A feline is sitting outdoors" | **~0.85** (high — synonymous meaning) |
| "The cat sits outside" | "A dog is playing in the garden" | ~0.28 (moderate — both outdoor animal scenes) |
| "The cat sits outside" | "The stock market crashed yesterday" | ~0.02 (near zero — unrelated topics) |

**Retrieval Behavior for "bank" query:**
- Treats each sentence as a single meaning unit — cannot disambiguate the two senses of "bank" within its holistic embedding.
- Sentences 3–5 (financial bank context) rank higher due to the deposit context dominating the sentence-level representation.

---

### 2. Sentence Transformer — Word Embeddings

**Model:** `all-MiniLM-L6-v2` (base transformer layer accessed via `model._modules['0'].model`) | **Embedding dim:** 384 | **Method:** Per-token `last_hidden_state`

**How it works:** Bypasses the pooling layer to extract token-level embeddings directly from the underlying MiniLM transformer.

**Cosine Similarity for "bank" (sentence 1, example):**

| Token A | Token B | Score |
|---|---|---|
| `[CLS]` | `bank` | ~0.30 |
| `The` | `bank` | ~0.25 |
| `bank` | `river` | ~0.55 |

**Retrieval Behavior for "bank" query:**
- Extracts the embedding of the token `bank` from the query, then searches FAISS for the closest `bank` token embeddings across documents.
- Captures partial contextual sensitivity (MiniLM is trained for sentence-level tasks, so word-level context is less refined than BERT).
- Dimension is **384** — consistent across both sentence and word uses of this model.

---

### 3. BERT — Sentence Embeddings

**Model:** `bert-base-cased` | **Embedding dim:** 768 | **Method:** `[CLS]` token embedding

**How it works:** BERT's special `[CLS]` token aggregates sequence-level context through bidirectional self-attention. Its final hidden state is used as the sentence embedding.

**Pairwise Cosine Similarities (`document_activity`):**

| Sentence A | Sentence B | Score |
|---|---|---|
| "The bank is near the river." | "The cat is near the bank." | ~0.97 |
| "The bank is near the river." | "Deposit 10000 RS in the bank." | ~0.90 |
| "Deposit 10000 RS in the bank near river." | "Deposit 10000 RS in the bank near river bank." | ~0.99 |

**Observation:** BERT sentence-level `[CLS]` embeddings cluster all sentences **very closely** (all scores >0.85), making them hard to distinguish at the sentence level. The model sees all sentences as contextually related because "bank" appears in all of them.

**Retrieval Result (Query: "What does bank mean?"):**

BERT's sentence embeddings return results ordered by general topical similarity rather than fine-grained semantic distinction between "river bank" and "financial bank."

---

### 4. BERT — Word Embeddings

**Model:** `bert-base-cased` | **Embedding dim:** 768 | **Method:** Per-token `last_hidden_state` for the word `bank`

**How it works:** BERT generates a unique embedding for every token based on its full bidirectional context. The word "bank" receives a different vector in each sentence it appears in — this is **contextual disambiguation**.

**Pairwise Cosine Similarity for "bank" token across sentences:**

| Context A | Context B | Score | Interpretation |
|---|---|---|---|
| "The **bank** is near the river." | "The cat is near the **bank**." | **0.9293** | Both likely = river bank |
| "Deposit 10000 RS in the **bank**." | "Deposit 10000 RS in the **bank** near river." | **0.9865** | Both = financial institution |
| "The **bank** is near the river." | "Deposit 10000 RS in the **bank**." | **0.7934** | Different senses — lower score |
| "The **bank** is near the river." | "river **bank**." (2nd occurrence in sent. 5) | **0.6795** | Most different — clearest disambiguation |

**Key finding:** BERT word embeddings successfully differentiate between the geographical and financial senses of "bank" through contextual variation — a clear advantage over static embeddings.

**Note on sentence 5:** `"Deposit 10000 RS in the bank near river bank."` contains the token `bank` **twice**, yielding **6 total "bank" embeddings** across the 5 sentences. FAISS indexes all 6 independently.

---

## Summary Comparison Table

| Dimension | ST — Sentence Emb. | ST — Word Emb. | BERT — Sentence Emb. | BERT — Word Emb. |
|---|---|---|---|---|
| **Embedding size** | 384 | 384 | 768 | 768 |
| **Granularity** | Sentence-level | Token-level | Sentence-level (CLS) | Token-level |
| **Disambiguation** | ❌ No | ⚠️ Partial | ❌ No | ✅ Yes |
| **Retrieval quality** | Good for topic-level | Moderate | High similarity but low contrast | Best for polysemy tasks |
| **Use case** | Semantic search, RAG | Token analysis | General NLP tasks | Word sense disambiguation |
| **Speed** | Fastest | Fast | Moderate | Moderate |

---

## Key Takeaways

- **Sentence embeddings** (both ST and BERT) are best for whole-document retrieval where holistic meaning matters. They are fast and effective for RAG pipelines.
- **Word embeddings** from BERT shine in **ambiguity resolution** — the same word ("bank") gets different embeddings depending on surrounding context, enabling finer retrieval.
- **SentenceTransformer** (`all-MiniLM-L6-v2`) is optimized for sentence-level tasks and outperforms raw BERT `[CLS]` embeddings for semantic similarity out of the box.
- **BERT word embeddings** are the most powerful for understanding polysemy, demonstrating BERT's core design principle of **bidirectional contextual encoding**.
- **FAISS (IndexFlatL2)** works effectively with all four embedding types — dimensions just need to match between index creation and query time.
