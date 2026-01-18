import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai
import numpy as np
import faiss
import pickle
import os

# -------------------------
# 1️⃣ SETUP STREAMLIT UI
# -------------------------

st.set_page_config(page_title="Simple RAG Chatbot", layout="wide")

st.title("📄 Simple PDF Chatbot (RAG)")
st.write("Upload a PDF, then ask questions about it.")

# -------------------------
# 2️⃣ GET API KEY
# -------------------------

api_key = st.text_input("Enter your Google API Key:", type="password")

if api_key:
    genai.configure(api_key=api_key)

# -------------------------
# 3️⃣ READ PDF FUNCTION
# -------------------------

def read_pdf(files):
    text = ""
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# -------------------------
# 4️⃣ SPLIT TEXT INTO CHUNKS
# -------------------------

def split_text(text):
    chunks = []
    words = text.split()
    chunk_size = 200

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks

# -------------------------
# 5️⃣ CREATE EMBEDDING
# -------------------------

def get_embedding(text):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return np.array(result["embedding"], dtype=np.float32)

# -------------------------
# 6️⃣ BUILD FAISS INDEX
# -------------------------

def build_index(chunks):
    vectors = np.array([get_embedding(chunk) for chunk in chunks])

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # Save index + text chunks
    faiss.write_index(index, "faiss_index.bin")
    with open("texts.pkl", "wb") as f:
        pickle.dump(chunks, f)

# -------------------------
# 7️⃣ SEARCH FUNCTION
# -------------------------

def search_index(query, k=5):
    index = faiss.read_index("faiss_index.bin")
    with open("texts.pkl", "rb") as f:
        texts = pickle.load(f)

    query_vector = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_vector, k)

    results = [texts[i] for i in indices[0]]
    return results

# -------------------------
# 8️⃣ ASK GEMINI WITH CONTEXT
# -------------------------

def ask_gemini(question, context):
    model = genai.GenerativeModel("gemini-3-flash-preview")

    prompt = f"""
    Answer the question based ONLY on this context:

    {context}

    Question: {question}
    """

    response = model.generate_content(prompt)
    return response.text

# -------------------------
# 9️⃣ MAIN APP LOGIC
# -------------------------

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload PDF")
    pdf_files = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True,
        type=["pdf"]
    )

    if st.button("Process PDFs"):
        if not api_key:
            st.error("Please enter API Key first!")
        elif not pdf_files:
            st.error("Please upload at least one PDF!")
        else:
            with st.spinner("Processing..."):
                text = read_pdf(pdf_files)
                chunks = split_text(text)
                build_index(chunks)
                st.success("PDF Processed Successfully!")

# User question input
question = st.text_input("Ask a question about your PDF:")

if st.button("Get Answer"):
    if not api_key:
        st.error("Enter API Key first!")
    elif not os.path.exists("faiss_index.bin"):
        st.error("Please upload and process PDF first!")
    elif not question:
        st.error("Please enter a question!")
    else:
        docs = search_index(question, k=5)
        context = "\n\n".join(docs)
        answer = ask_gemini(question, context)

        st.subheader("Answer:")
        st.write(answer)
