# === Upgraded Groq PDF Chatbot ===
import streamlit as st
import requests
import fitz  # PyMuPDF
import os
import tempfile
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# === Config ===
st.set_page_config(page_title="Groq PDF Chatbot", page_icon="🤖")

# === Secret API Key ===
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# === Session State ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# === Embedding and Vector DB Setup ===
@st.cache_resource
def get_vectorstore(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(texts, embedding=embeddings)

# === PDF Extraction ===
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# === Summarize ===
def summarize(text):
    summary_prompt = [
        {"role": "system", "content": "Summarize this document in 5 bullet points."},
        {"role": "user", "content": text[:4000]}
    ]
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": "llama3-8b-8192", "messages": summary_prompt}
    )
    return response.json()["choices"][0]["message"]["content"] if response.ok else "(Could not summarize)"

# === UI ===
st.markdown("<h1 style='text-align: center;'>🤖 Groq Chatbot with PDF by Viraj</h1>", unsafe_allow_html=True)
st.sidebar.header("📎 Upload a PDF")
uploaded_pdf = st.sidebar.file_uploader("Upload", type="pdf")

vectorstore = None
if uploaded_pdf:
    with st.spinner("Extracting & indexing text..."):
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        vectorstore = get_vectorstore(pdf_text)
        st.sidebar.success("✅ PDF processed!")
        summary = summarize(pdf_text)
        st.markdown("**📄 PDF Summary:**")
        st.markdown(summary)

# === Chat Interface ===
st.markdown("<hr>", unsafe_allow_html=True)
user_input = st.chat_input("Ask something about the PDF...")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # === RAG-style context ===
    context = ""
    if vectorstore:
        docs = vectorstore.similarity_search(user_input, k=3)
        context = "\n---\n".join([doc.page_content for doc in docs])

    system_msg = {
        "role": "system",
        "content": f"You are a helpful assistant using the following document context:\n{context}"
    }

    full_messages = [system_msg] + st.session_state.messages[-10:]  # limit context

    with st.spinner("Thinking..."):
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama3-8b-8192", "messages": full_messages}
        )

    if response.ok:
        reply = response.json()["choices"][0]["message"]["content"]
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        st.error("⚠️ Something went wrong.")
        st.code(response.text, language="json")

# === Clear Chat ===
if st.sidebar.button("🔁 Clear Chat"):
    st.session_state.messages.clear()
