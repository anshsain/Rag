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
- Persist vector DB
- File uploads
- Automated eval pipelines
- Token / cost dashboards
- Multi-document ingestion

## Example Questions & Answers

The following examples demonstrate the behavior of the RAG system using an ingested document.

### Q1
**Question:** What is the capital of India?  
**Answer:** New Delhi is the capital of India.  
**Source:** Chunk containing “The capital of India is New Delhi.”

---

### Q2
**Question:** Is India located in Europe?  
**Answer:** I don’t know based on the provided context.  
**Explanation:** The ingested document does not mention Europe.

---

### Q3
**Question:** What is India’s financial capital?  
**Answer:** Mumbai is the financial capital of India.  
**Source:** Chunk mentioning Mumbai.

---

### Q4
**Question:** What continent is India part of?  
**Answer:** India is part of South Asia.  
**Source:** Chunk describing India’s geographic location.

---

### Q5
**Question:** What is the population of India?  
**Answer:** I don’t know based on the provided context.  
**Explanation:** Population data was not present in the ingested document.

## Short Note on System Behavior and Success Criteria

The success of this RAG system is evaluated qualitatively rather than through formal precision or recall metrics.

A response is considered successful if the answer is fully grounded in the retrieved context and correctly addresses the user’s query. When the retrieved documents do not contain sufficient information, the system explicitly responds with “I don’t know,” avoiding hallucinations.

This behavior demonstrates correct RAG principles: retrieval-driven grounding, controlled generation, and graceful failure in the absence of evidence. The system prioritizes factual correctness and transparency over completeness.

Given the limited scope and single-document ingestion setup, this approach provides a clear and reliable demonstration of retrieval-augmented generation.
