import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate

import os

# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
<style>
#normal-blink {
    text-align: center;
    background: rgba(0, 0, 0, 0.75);
    padding: 5px 30px;
    border-radius: 12px;
    border: 3px solid #f28705;
    backdrop-filter: blur(5px);
    margin-top: 20px;
}
.blink {
    font-size: 35px;
    font-weight: bold;
    color: #00C4FF;
    animation: blinker 1s linear infinite;
}
@keyframes blinker {
    50% { opacity: 0; }
}
</style>

<div id="normal-blink">
    <div class="blink">Welcome To My Chatbot 💁</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
## Document Genie: Get instant insights from your Documents

This chatbot uses Retrieval-Augmented Generation (RAG) with Google's **Gemini-2.5-flash** model.
It:
1) Reads PDFs  
2) Splits them into chunks  
3) Creates embeddings  
4) Stores them in FAISS  
5) Retrieves relevant chunks  
6) Generates accurate answers  

### Steps:
1. Enter your Google API Key  
2. Upload PDFs and click **Submit & Process**  
3. Ask questions about your documents  
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
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_store")

def get_qa_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    document_chain = create_stuff_documents_chain(model, prompt)
    return document_chain

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

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    document_chain = get_qa_chain(api_key)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({
        "input": user_question
    })

    st.write("Reply:")
    st.write(response["answer"])

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
