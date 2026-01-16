# Mini RAG Application

A minimal **Retrieval-Augmented Generation (RAG)** application built using **Streamlit**.  
Users can ingest text, store embeddings in a **hosted vector database (Pinecone)**, retrieve relevant chunks, rerank them, and generate grounded answers using an LLM with citations.

---

**Live URL:**  
https://6qcopzzur5b3eo5j5tymfu.streamlit.app/

**Resume Link:**  
https://drive.google.com/file/d/1s0uzvob_2NpCEj2M-V6NPttDjTFw9m9C/view?usp=sharing

---

## Architecture

User  
→ Ingest Text  
→ Chunking (800 tokens, 100 overlap)  
→ Embeddings (MiniLM, 384-dim)  
→ Hosted Vector Database (Pinecone)  
→ Query  
→ Vector Retrieval (Top-K = 8)  
→ Reranking (Cohere)  
→ LLM Answer (Groq)  
→ Citations  

---

## Tech Stack

| Component | Tool |
|---------|------|
| Frontend | Streamlit |
| Embeddings | sentence-transformers/paraphrase-MiniLM-L3-v2 |
| Vector Database | Pinecone (hosted, dense vectors) |
| LLM | Groq (LLaMA 3.1 8B Instant) |
| Reranker | Cohere Rerank v3 |
| Chunking / LLM Orchestration | LangChain (non-vector components) |

---

## Chunking Strategy

- **Chunk size:** 800 tokens  
- **Overlap:** 100 tokens  

This preserves semantic continuity while maintaining retrieval precision.

---

## Retrieval & Reranking

### Vector Retrieval
- Dense vector embeddings (384 dimensions)
- Cosine similarity search in Pinecone
- Top-K = 8 retrieved chunks

### Reranking
- Model: `rerank-english-v3.0` (Cohere)
- Top-N = 3 chunks

Reranking improves relevance before passing context to the LLM.

---

## Answer Generation

- The LLM is instructed to:
  - Use **ONLY the retrieved context**
  - Respond with **“I don’t know”** if the answer is not present
- Answers are returned with **inline citations** mapped to source chunks

This ensures grounded, non-hallucinated responses.

---

## Minimal Evaluation (Manual)

| Question | Expected Answer | Result |
|--------|----------------|--------|
| What is Retrieval-Augmented Generation? | Grounded definition | ✅ Correct |
| What is the capital of France? | I don’t know | ✅ Correct |

---

## Example Questions & Answers

### Q1
**Question:** What is Retrieval-Augmented Generation?  
**Answer:** Retrieval-Augmented Generation is a technique that combines external knowledge retrieval with language model generation to produce grounded answers.  
**Source:** Chunk describing RAG definition.

### Q2
**Question:** Why is reranking used in this system?  
**Answer:** Reranking improves the relevance of retrieved document chunks before generating the final answer.  
**Source:** Chunk explaining reranking.

### Q3
**Question:** What role does the vector database play?  
**Answer:** The vector database stores document embeddings and enables similarity-based retrieval.  
**Source:** Chunk describing vector storage.

### Q4
**Question:** What is the capital of France?  
**Answer:** I don’t know based on the provided context.  
**Explanation:** The ingested document does not contain this information.

### Q5
**Question:** Does this system hallucinate answers?  
**Answer:** No. The system explicitly refuses to answer when the information is not present in the retrieved context.

---

## Limitations & Trade-offs

- Single-document ingestion (text paste only)
- No file upload support
- No persistent evaluation pipeline
- Basic cost and token tracking
- Manual evaluation instead of automated metrics

---

## What I’d Do Next

- Add PDF and multi-file uploads
- Support multi-document ingestion
- Add automated evaluation (precision/recall)
- Implement token and cost dashboards
- Add query history and session memory

---

## Short Note on System Behavior & Success Criteria

The system is evaluated qualitatively rather than through formal precision or recall metrics.

A response is considered successful if it is fully grounded in the retrieved context and correctly answers the query.  
When relevant information is not present, the system explicitly responds with **“I don’t know”**, avoiding hallucinations.

This demonstrates correct RAG behavior: retrieval-driven grounding, controlled generation, and graceful failure.
