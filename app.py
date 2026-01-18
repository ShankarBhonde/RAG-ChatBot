import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai
import numpy as np
import faiss
import pickle
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Document Genie: Get instant insights from your Documents
""")

# ---------------- API KEY INPUT ---------------- #

api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

if api_key:
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"API Key configuration failed: {str(e)}")

# ---------------- HELPER FUNCTIONS ---------------- #

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)

def get_embedding(text):
    """Generate embedding using Google Gemini directly (NO LANGCHAIN)"""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return np.array(result["embedding"], dtype=np.float32)

def build_faiss_index(chunks):
    """Create FAISS index manually (this is the part that fixes your error)"""
    vectors = np.array([get_embedding(chunk) for chunk in chunks])

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # Save index + text chunks
    faiss.write_index(index, "faiss_index.bin")
    with open("faiss_texts.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_faiss_index():
    index = faiss.read_index("faiss_index.bin")
    with open("faiss_texts.pkl", "rb") as f:
        texts = pickle.load(f)
    return index, texts

def search_faiss(query, k=5):
    index, texts = load_faiss_index()
    query_vector = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    results = [texts[i] for i in indices[0]]
    return results

def ask_gemini_with_context(question, docs):
    """Uses your model: gemini-3-flash-preview"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.3,
        google_api_key=api_key
    )

    context = "\n\n".join(docs)

    prompt = f"""
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    return llm.invoke(prompt).content

# ---------------- MAIN APP ---------------- #

def main():
    st.header("AI Clone Chatbot 💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")
    ask_btn = st.button("Submit Question")

    if ask_btn:
        if not api_key:
            st.error("⚠ Please enter your API Key first.")
        elif not user_question:
            st.error("⚠ Please enter a question.")
        elif not os.path.exists("faiss_index.bin"):
            st.error("⚠ Please upload and process PDF first.")
        else:
            docs = search_faiss(user_question, k=5)
            answer = ask_gemini_with_context(user_question, docs)

            st.write("Reply:")
            st.write(answer)

    with st.sidebar:
        st.title("Menu:")

        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on Submit & Process",
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        if st.button("Submit & Process", key="process_button"):
            if not api_key:
                st.error("⚠ Please enter your API Key first.")
            elif not pdf_docs:
                st.error("⚠ Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    build_faiss_index(text_chunks)
                    st.success("✅ PDFs Processed & FAISS index created!")

if __name__ == "__main__":
    main()
