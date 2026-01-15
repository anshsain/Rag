import streamlit as st
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

# ------------------ CONFIG ------------------

st.set_page_config(page_title="Mini RAG", layout="centered")
st.title("📄 Mini RAG Application")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ------------------ INIT ------------------

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

vectorstore = Qdrant(
    client,
    collection_name="mini_rag_docs",
    embeddings=embeddings,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

# ------------------ INGEST ------------------

st.subheader("Ingest Document")

text = st.text_area("Paste text to ingest")

if st.button("Ingest"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    docs = splitter.create_documents([text])
    vectorstore.add_documents(docs)
    st.success(f"Ingested {len(docs)} chunks")

# ------------------ QUERY ------------------

st.subheader("Ask a Question")

question = st.text_input("Your question")

if st.button("Ask"):
    docs = vectorstore.similarity_search(question, k=3)

    if not docs:
        st.warning("No relevant context found.")
    else:
        context = "\n\n".join([d.page_content for d in docs])

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
