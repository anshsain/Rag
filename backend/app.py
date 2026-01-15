import streamlit as st
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ------------------ PAGE CONFIG ------------------

st.set_page_config(page_title="Mini RAG", layout="centered")
st.title("Mini RAG Application")

# ------------------ ENV VARS ------------------

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY or not GEMINI_API_KEY:
    st.error("Missing environment variables. Check QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY.")
    st.stop()

# ------------------ INIT ------------------

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

COLLECTION_NAME = "mini_rag_docs"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

COLLECTION_NAME = "mini_rag_docs"

existing_collections = [
    c.name for c in client.get_collections().collections
]

if COLLECTION_NAME not in existing_collections:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,  # MiniLM embedding size
            distance=Distance.COSINE,
        ),
    )


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

vectorstore = Qdrant(
    client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

# ------------------ RERANKER ------------------

def rerank_documents(question, docs, llm, top_n=3):
    scored_docs = []

    for doc in docs:
        prompt = f"""
Score how relevant the following passage is to the question.
Return ONLY a number between 0 and 10.

Question:
{question}

Passage:
{doc.page_content}
"""
        try:
            score = float(llm.invoke(prompt).content.strip())
        except:
            score = 0.0

        scored_docs.append((doc, score))

    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_n]]

# ------------------ INGEST ------------------

st.subheader("Ingest Document")

text = st.text_area("Paste text to ingest")

if st.button("Ingest"):
    if not text.strip():
        st.warning("Please paste some text.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100
    )

    docs = splitter.create_documents(
        [text],
        metadatas=[{"source": "user_input"}]
    )

    vectorstore.add_documents(docs)
    st.success(f"Ingested {len(docs)} chunks")

# ------------------ QUERY ------------------

st.subheader(" Ask a Question")

question = st.text_input("Your question")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Retrieve
    candidate_docs = vectorstore.similarity_search(question, k=8)

    if not candidate_docs:
        st.warning("No relevant context found.")
        st.stop()

    # Rerank
    docs = rerank_documents(question, candidate_docs, llm, top_n=3)

    # Build context with citations
    context = "\n\n".join(
        [f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)]
    )

    prompt = f"""
Answer the question using ONLY the context below.
Add inline citations like [1], [2] after each factual statement.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{question}
"""

    # LLM Answer
    response = llm.invoke(prompt)

    st.markdown("### Answer")
    st.write(response.content)

    # Sources
    st.markdown("### Sources")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        st.write(f"[{i+1}] Source: {source}")
