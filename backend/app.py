import streamlit as st
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# ------------------ CONFIG ------------------

st.set_page_config(page_title="Mini RAG", layout="centered")
st.title("Mini RAG Application")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COLLECTION_NAME = "mini_rag_docs"
EMBEDDING_DIM = 384  # MiniLM-L3-v2

if not QDRANT_URL or not QDRANT_API_KEY or not GEMINI_API_KEY:
    st.error("❌ Missing environment variables")
    st.stop()

# ------------------ QDRANT CLIENT ------------------

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# ------------------ FORCE CREATE COLLECTION ------------------
# ⚠️ This is the KEY FIX

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_DIM,
        distance=Distance.COSINE,
    ),
)

# ------------------ EMBEDDINGS ------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

# ------------------ VECTOR STORE ------------------

vectorstore = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
)

# ------------------ LLM ------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)

# ------------------ INGEST ------------------

st.subheader("Ingest Document")

text = st.text_area("Paste text to ingest")

if st.button("Ingest"):
    if not text.strip():
        st.warning("Please paste some text.")
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )
        docs = splitter.create_documents([text])
        vectorstore.add_documents(docs)
        st.success(f"✅ Ingested {len(docs)} chunks")

# ------------------ QUERY ------------------

st.subheader("Ask a Question")

question = st.text_input("Your question")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        docs = vectorstore.similarity_search(question, k=3)

        if not docs:
            st.warning("No relevant context found.")
        else:
            context = "\n\n".join(d.page_content for d in docs)

            prompt = f"""
Use ONLY the context below to answer.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""

            response = llm.invoke(prompt)
            st.markdown("### ✅ Answer")
            st.write(response.content)
