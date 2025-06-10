import streamlit as st
import requests
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Load API Key
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Embedder
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Split text into chunks
def split_text(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.create_documents([text])
    return chunks

# Embed chunks - Only accept list of strings (hashable)
@st.cache_resource
def embed_text_chunks(text_list):
    return model.encode(text_list, convert_to_tensor=True)

# Get top-k similar chunks
def get_similar_chunks(query, text_list, embeddings, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)[0]
    return [text_list[hit["corpus_id"]] for hit in hits]

# Streamlit App UI
st.set_page_config(page_title="Chatbot using Groq", page_icon="🧠")
st.markdown("<h1 style='text-align: center;'>🤖 Groq Chatbot by Viraj</h1>", unsafe_allow_html=True)
st.markdown("<hr style='margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# Session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# PDF Upload Sidebar
st.sidebar.header("📎 Upload a PDF to Chat With It")
uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type="pdf")

pdf_text = ""
chunks = []
text_list = []
embeddings = None

if uploaded_pdf:
    pdf_text = extract_text_from_pdf(uploaded_pdf)
    chunks = split_text(pdf_text)
    text_list = [chunk.page_content for chunk in chunks]  # ✅ Extract only text
    embeddings = embed_text_chunks(text_list)
    st.sidebar.success("✅ PDF uploaded and processed!")

# Display previous messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div style='text-align: right; background-color: #dcf8c6; padding: 10px 15px; margin: 10px 0; border-radius: 12px; max-width: 80%; margin-left: auto;'>{msg['content']}</div>",
            unsafe_allow_html=True
        )
    elif msg["role"] == "assistant":
        st.markdown(
            f"<div style='text-align: left; background-color: #f1f0f0; padding: 10px 15px; margin: 10px 0; border-radius: 12px; max-width: 80%; margin-right: auto;'>{msg['content']}</div>",
            unsafe_allow_html=True
        )

# Chat input
prompt = st.chat_input("Ask me anything!")

if prompt:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # If PDF is uploaded and embeddings are available, enhance context
    if uploaded_pdf and embeddings is not None:
        similar_chunks = get_similar_chunks(prompt, text_list, embeddings)
        context = "\n\n".join(similar_chunks)
        st.session_state.messages.insert(0, {
            "role": "system",
            "content": f"Use this PDF content to answer queries:\n\n{context}"
        })

    # Make API call to Groq
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": "llama3-8b-8192",
        "messages": st.session_state.messages
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=body,
    )

    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"]
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").markdown(reply)
    else:
        st.error("❌ API Error: " + response.text)
