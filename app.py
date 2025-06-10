import streamlit as st
import fitz  # PyMuPDF
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests

# Load API key from Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Extract text from uploaded PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Create vector store using Chroma
def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./chroma_store")
    return vectorstore

# Retrieve context from vector store
def get_relevant_context(vectorstore, query, k=4):
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

# Streamlit Page Config
st.set_page_config(page_title="Groq PDF Chatbot", page_icon="📄")
st.markdown("<h1 style='text-align: center;'>📄 Chat with PDF using GROQ</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Session State for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for uploading PDF
st.sidebar.header("📎 Upload a PDF")
pdf_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

vectorstore = None
if pdf_file:
    pdf_text = extract_text_from_pdf(pdf_file)
    vectorstore = create_vector_store(pdf_text)
    st.sidebar.success("✅ PDF successfully processed!")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask something about your PDF...")

if user_input and vectorstore:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Retrieve context
    context = get_relevant_context(vectorstore, user_input)

    # Prepare GROQ API call
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": f"Use the following PDF context to answer the question:\n\n{context}"},
            {"role": "user", "content": user_input}
        ]
    }

    # Call Groq API
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body)

    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("❌ Error from Groq API")
        st.code(response.text, language="json")

# Clear chat
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages.clear()
    st.experimental_rerun()
