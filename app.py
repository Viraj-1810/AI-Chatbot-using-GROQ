import streamlit as st
import fitz  # PyMuPDF
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util

# Load API key
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize embedding model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from PDF
def extract_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# Chunk text into documents
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.create_documents([text])

# Compute embeddings for chunks
@st.cache_resource
def embed_chunks(chunks):
    model = get_embedding_model()
    texts = [doc.page_content for doc in chunks]
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

# Retrieve context via cosine similarity
def get_context(chunks, embeddings, question, top_k=4):
    model = get_embedding_model()
    q_embed = model.encode(question, convert_to_tensor=True)
    cos_scores = util.semantic_search(q_embed, embeddings, top_k=top_k)[0]
    selected = [chunks[hit["corpus_id"]].page_content for hit in cos_scores]
    return "\n\n".join(selected)

# Streamlit UI
st.set_page_config(page_title="Groq PDF Chatbot", page_icon="🔍")
st.title("🔍 Chat with PDF via GROQ")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded = st.sidebar.file_uploader("Upload PDF", type="pdf")
chunks = embeddings = None

if uploaded:
    text = extract_text(uploaded)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    st.sidebar.success("✅ PDF ready!")

for msg in st.session_state.history:
    st.chat_message(msg["role"]).markdown(msg["content"])

prompt = st.chat_input("Ask anything about the PDF...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})

    if chunks and embeddings is not None:
        context = get_context(chunks, embeddings, prompt)
    else:
        context = ""

    body = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": f"Use this context:\n\n{context}"},
            {"role": "user", "content": prompt}
        ]
    }
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json=body
    )

    if resp.status_code == 200:
        reply = resp.json()["choices"][0]["message"]["content"]
        st.chat_message("assistant").markdown(reply)
        st.session_state.history.append({"role": "assistant", "content": reply})
    else:
        st.error("❌ GROQ API error.")
        st.code(resp.text, language="json")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.history.clear()
    st.experimental_rerun()
