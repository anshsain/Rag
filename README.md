# Mini RAG Application 

A minimal **Retrieval-Augmented Generation (RAG)** application built using **Streamlit**.  
Users can ingest text, store embeddings in a vector database, retrieve relevant chunks, rerank them, and generate grounded answers using an LLM with citations.

---

**Live URL:** <https://6qcopzzur5b3eo5j5tymfu.streamlit.app/>
**Resume Link:** <https://drive.google.com/file/d/1s0uzvob_2NpCEj2M-V6NPttDjTFw9m9C/view?usp=sharing> 
---

## Architecture
User -> Ingest Text -> Chunking (800 tokens, 100 overlap) -> Embeddings (MiniLM) -> Vector Store (Chroma) -> Ask Question -> Vector Retrieval (Top-K) -> Reranking (Cohere) -> LLM Answer (Groq) -> Citations

---

## Tech Stack

| Component        | Tool |
|-----------------|------|
| Frontend        | Streamlit |
| Embeddings      | `sentence-transformers/paraphrase-MiniLM-L3-v2` |
| Vector Database | Chroma |
| LLM             | Groq (LLaMA 3.x) |
| Reranker        | Cohere Rerank v3 |
| Framework       | LangChain |

---

## Chunking Strategy

- **Chunk size:** 800 tokens  
- **Overlap:** 100 tokens  
- **Why:** Preserves semantic continuity while maintaining retrieval precision

---

## Retrieval & Reranking

1. **Initial retrieval**
   - Cosine similarity search
   - Top-K = 8

2. **Reranking**
   - Cohere Rerank (`rerank-english-v3.0`)
   - Top-N = 3

This improves relevance before passing context to the LLM.

---

## Answer Generation

- The LLM is instructed to:
  - Use **ONLY the retrieved context**
  - Say **“I don’t know”** if the answer is not present
- Answers are displayed with **citations** mapped to source chunks

---

## Minimal Evaluation

| Question | Expected Answer | Result |
|--------|----------------|--------|
| What is the capital of India? | New Delhi | ✅ Correct |
| Is India in Europe? | I don’t know | ✅ Correct |

## Limitations & Trade-offs

- In-memory vector store (data resets on reload)
- No document upload (text paste only)
- No persistent evaluation pipeline
- Limited cost / token tracking

## What I’d Do Next

-Persist vector DB
- File uploads
- Automated eval pipelines
- Token / cost dashboards
- Multi-document ingestion
