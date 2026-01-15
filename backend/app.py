import streamlit as st
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

# ------------------ CONFIG ------------------

st.set_page_config(page_title="Mini RAG", layout="centered")
st.title("📄 Mini RAG Application")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not set")
    st.stop()

# ------------------ EMBEDDINGS ------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

# ------------------ ONE-TIME DOCUMENT ------------------

base_text = """
India is a country in South Asia.
The capital of India is New Delhi.
New Delhi is located in the northern part of India.
India follows a parliamentary democratic system.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

docs = splitter.create_documents([base_text])

# ------------------ QDRANT (IN-MEMORY, CORRECT WAY) ------------------
# 🔥 THIS IS THE KEY FIX

client = QdrantClient(":memory:")

vectorstore = Qdrant.from_documents(
    docs,
    embedding=embeddings,
    client=client,
    collection_name="mini_rag_docs",
)

# ------------------ LLM ------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)

# ------------------ QUERY ------------------

st.subheader("❓ Ask a Question")

question = st.text_input("Your question")

if st.button("Ask") and question:
    docs = vectorstore.similarity_search(question, k=3)

    if not docs:
        st.warning("No relevant context found.")
    else:
        context = "\n\n".join(
            [f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)]
        )

        prompt = f"""
Use ONLY the context below to answer.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer with citations like [1], [2].
"""

        response = llm.invoke(prompt)

        st.markdown("### ✅ Answer")
        st.write(response.content)

        st.markdown("### 📚 Sources")
        for i, doc in enumerate(docs):
            st.write(f"[{i+1}] {doc.page_content}")
