import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

import os

# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Document Genie: Get instant insights from your Documents
""")

# ---------------- API KEY INPUT ---------------- #

api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

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

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_store")   # Auto-creates folder

def ask_gemini_with_context(question, docs, api_key):
    """
    Simple, reliable way to ask Gemini with retrieved context
    (no fragile LangChain chains)
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    # Combine retrieved documents into one context string
    context = "\n\n".join([doc.page_content for doc in docs])

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

def user_input(user_question, api_key):
    if not os.path.exists("faiss_store"):
        st.error("⚠ Please upload and process PDF first.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )

    vector_db = FAISS.load_local(
        "faiss_store",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retrieve top 5 relevant chunks
    docs = vector_db.similarity_search(user_question, k=5)

    answer = ask_gemini_with_context(user_question, docs, api_key)

    st.write("Reply:")
    st.write(answer)

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
        else:
            user_input(user_question, api_key)

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
                    get_vector_store(text_chunks, api_key)
                    st.success("✅ PDFs Processed Successfully!")

if __name__ == "__main__":
    main()
