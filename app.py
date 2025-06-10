import streamlit as st
import requests
import fitz  # PyMuPDF

# Load API Key securely
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# PDF Extractor
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# App UI
st.set_page_config(page_title="Groq PDF Chatbot", page_icon="🤖")
st.markdown("<h1 style='text-align: center;'>📄 PDF Chatbot using Groq</h1>", unsafe_allow_html=True)

# Session for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF upload
st.sidebar.header("📎 Upload PDF")
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

if uploaded_pdf:
    pdf_text = extract_text_from_pdf(uploaded_pdf)
    st.sidebar.success("✅ PDF uploaded successfully!")
    # Store content in system message (cut to 3000 chars)
    if not any(msg["role"] == "system" for msg in st.session_state.messages):
        st.session_state.messages.insert(0, {
            "role": "system",
            "content": f"The following is the extracted text from the uploaded PDF. Use this content to answer questions:\n\n{pdf_text[:3000]}"
        })

# Display past messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div style='text-align: right; background-color: #dcf8c6; padding: 10px; border-radius: 10px; max-width: 80%; margin-left: auto;'>{msg['content']}</div>", unsafe_allow_html=True)
    elif msg["role"] == "assistant":
        st.markdown(f"<div style='text-align: left; background-color: #f1f0f0; padding: 10px; border-radius: 10px; max-width: 80%; margin-right: auto;'>{msg['content']}</div>", unsafe_allow_html=True)

# User input
prompt = st.chat_input("Ask something about the PDF or anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Send request to Groq API
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama3-8b-8192",
        "messages": st.session_state.messages
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"]
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").markdown(reply)
    else:
        st.error("❌ API error occurred.")
        st.code(response.text, language="json")
