import streamlit as st
import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests
import os


load_dotenv()
# --- AWS & Bedrock Configuration ---
# Use an environment variable to specify the AWS profile name
# Example: export AWS_PROFILE=my-bedrock-profile
aws_profile_name = os.getenv("AWS_PROFILE")
if not aws_profile_name:
    st.error("Please set the AWS_PROFILE environment variable.")
    st.stop()

# Initialize Bedrock clients using the specified AWS profile
try:
    boto3_session = boto3.Session(profile_name=aws_profile_name)
    bedrock_llm_client = boto3_session.client("bedrock-runtime")
    bedrock_emb_client = boto3_session.client("bedrock-runtime")
except Exception as e:
    st.error(f"Failed to create Boto3 session with profile '{aws_profile_name}': {e}")
    st.stop()

# Initialize Bedrock models
bedrock_llm = ChatBedrock(
    model_id="amazon.titan-text-express-v1",
    client=bedrock_llm_client,
    model_kwargs={"temperature": 0.3}
)
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_emb_client
)

# --- Helper Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_webpage_text(urls):
    """Extracts text from a list of web page URLs."""
    text = ""
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract only the text content
            text += soup.get_text(separator=' ', strip=True)
        except requests.exceptions.RequestException as e:
            st.warning(f"Could not retrieve content from {url}: {e}")
    return text

def get_text_chunks(text):
    """Splits a large text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, embeddings_model):
    """Creates a vector store from text chunks."""
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings_model)
    return vector_store

def get_qa_chain(vector_store, llm_model):
    """Defines the RAG conversational chain."""
    prompt_template = """
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])}
    )
    return qa_chain

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Tool", layout="wide")
st.title("RAG Tool: Answer Questions over Docs & Web 📚🔗")

# Sidebar for data ingestion
with st.sidebar:
    st.title("Data Ingestion")
    pdf_docs = st.file_uploader("Upload PDF Documents", accept_multiple_files=True, type=['pdf'])
    web_links = st.text_area("Enter Web Page URLs (one per line)")
    
    if st.button("Process Documents"):
        with st.spinner("Processing..."):
            raw_text = ""
            if pdf_docs:
                raw_text += get_pdf_text(pdf_docs)
            if web_links:
                urls = web_links.splitlines()
                raw_text += get_webpage_text(urls)
            
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks, bedrock_embeddings)
                st.session_state.vector_store = vector_store
                st.success("Documents processed successfully!")
            else:
                st.warning("Please upload at least one PDF or provide a web link.")

# Main content area for questions and answers
if "vector_store" in st.session_state:
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        with st.spinner("Searching for an answer..."):
            qa_chain = get_qa_chain(st.session_state.vector_store, bedrock_llm)
            response = qa_chain({"query": user_question})
            
            st.markdown("### Answer")
            st.write(response["result"])

            st.markdown("---")
            st.write("Source Documents:")
            for doc in response["source_documents"]:
                st.write(f"- Source: {doc.metadata.get('source', 'N/A')}")
else:
    st.info("Please upload documents or provide web links in the sidebar to get started.")