import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import tempfile

# Load env variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.title("📄 AskMyPDF — Your KnowItAll Document Assistant")
st.write("Upload a PDF, I’ll ingest it and you can ask questions about it.")

# Upload PDF
pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if pdf_file:
    # Save PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        pdf_path = tmp.name

    st.success("✅ PDF uploaded. Processing...")

    # Load PDF pages
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    # Embed + store in Chroma
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_store")
    vectordb.persist()

    st.success("✅ PDF ingested into ChromaDB. You can now ask questions!")

    # Create retriever
    retriever = vectordb.as_retriever()

    # Initialize GROQ LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192",  
        temperature=0
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    # User query
    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("🤔 Thinking..."):
            result = qa_chain(query)
            st.markdown("### 📖 Answer:")
            st.write(result['result'])