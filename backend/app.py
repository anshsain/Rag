import os
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

# ------------------ CONFIG ------------------

st.set_page_config(page_title="Mini RAG", layout="centered")
st.title("Mini RAG Application")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY or not GEMINI_API_KEY:
    st.error("❌ Environment variables not set")
    st.stop()

COLLECTION_NAME = "mini_rag_docs"

# ------------------ INIT CLIENTS ------------------

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

vectorstore = Qdrant(
    client=client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

# ------------------ ONE-TIME INGEST (SAFE) ------------------

try:
    existing = vectorstore.similarity_search("test", k=1)
except Exception:
    existing = []

if not existing:
    base_text = """
India is a country in South Asia.
The capital of India is New Delhi.
New Delhi is located in the northern part of India.
India has a parliamentary democratic system.
"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = splitter.create_documents([base_text])
    vectorstore.add_documents(docs)

# ------------------ QUERY ------------------

st.subheader("Ask a Question")

question = st.text_input("Your question")

if st.button("Ask") and question:
    docs = vectorstore.similarity_search(question, k=3)

    if not docs:
        st.warning("No relevant context found.")
        st.stop()

    context = "\n\n".join(
        [f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)]
    )

    prompt = f"""
Use ONLY the context below to answer the question.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{question}

Answer with citations like [1], [2].
"""

    response = llm.invoke(prompt)

    st.markdown("### Answer")
    st.write(response.content)

    st.markdown("### Sources")
    for i, doc in enumerate(docs):
        st.markdown(f"**[{i+1}]** {doc.page_content}")
