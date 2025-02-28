import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """Extract text from multiple PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text

def get_text_chunks(text):
    """Optimized text chunking for faster retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Optimized FAISS vector store creation."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(retriever):
    """Set up RetrievalQA pipeline."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context.".
    Do not provide incorrect answers.

    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)  # Optimized model

    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,  # Disable full doc retrieval for speed
        chain_type_kwargs={"prompt": prompt}
    )

    return chain

def user_input(user_question):
    """Process user input efficiently."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs={"k": 3})  # Limit search to 3 docs

    chain = get_conversational_chain(retriever)

    response = chain.invoke({"query": user_question})

    print(response)
    st.write("Reply: ", response["result"])  # Updated key for response

def main():
    """Streamlit UI."""
    st.set_page_config("Multiple PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF 📚 - Chat Agent 🤖")

    user_question = st.text_input("Ask a question about the uploaded PDFs .. ✍️📝")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📁 PDF Upload Section")
        pdf_docs = st.file_uploader(
            "Upload your PDFs & Click 'Submit & Process'",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):  # User-friendly loading message
                raw_text = get_pdf_text(pdf_docs)  # Extract text from PDFs
                text_chunks = get_text_chunks(raw_text)  # Split text
                get_vector_store(text_chunks)  # Create vector store
                st.success("Processing Complete! ✅")

if __name__ == "__main__":
    main()
