import os
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# ------------------ CONFIG ------------------

st.set_page_config(page_title="Mini RAG", layout="centered")
st.title("📄 Mini RAG Application")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not set")
    st.stop()

# ------------------ INIT (RUN ONCE) ------------------

if "vectorstore" not in st.session_state:

    client = QdrantClient(":memory:")

    COLLECTION_NAME = "mini_rag_docs"
    VECTOR_SIZE = 384

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

    vectorstore = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )

    COLLECTION_NAME = "mini_rag_docs"

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )

    # ---- INGEST ONCE ----
    if st.button("Ingest"):
    if not text.strip():
        st.warning("Please paste some text to ingest.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = splitter.create_documents([text])
    st.session_state.vectorstore.add_documents(docs)
    st.session_state.has_data = True

    st.success(f"Ingested {len(docs)} chunks")

if "has_data" not in st.session_state:
    st.session_state.has_data = False

# ------------------ LLM ------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)

# ------------------ QUERY ------------------

st.subheader("❓ Ask a Question")

question = st.text_input("Your question")
if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    if not st.session_state.has_data:
        st.warning("Please ingest a document first.")
        st.stop()

    vectorstore = st.session_state.vectorstore
    docs = vectorstore.similarity_search(question, k=3)

    if not docs:
        st.warning("No relevant context found.")
        st.stop()

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
"""

    response = llm.invoke(prompt)

    st.markdown("### ✅ Answer")
    st.write(response.content)

    st.markdown("### 📚 Sources")
    for i, doc in enumerate(docs):
        st.markdown(f"[{i+1}] {doc.page_content[:200]}...")

