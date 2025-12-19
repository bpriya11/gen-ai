RAG Tool 📚🔗
This project is a simple, plain-UI RAG (Retrieval-Augmented Generation) tool built with Streamlit and LangChain. It leverages Amazon Bedrock's powerful models to answer questions over user-provided documents and external web pages.

Features ✨
    PDF & Web Page Ingestion: Upload PDF files or provide web page URLs to use as a knowledge base.

    Context-Aware Q&A: Ask questions and get answers grounded in the provided documents.

    Bedrock Integration: Utilizes Amazon Bedrock models for both embeddings (amazon.titan-embed-text-v1) and text generation (amazon.titan-text-express-v1).

    Simple Streamlit UI: An intuitive web interface for a smooth user experience.

    AWS Profile Authentication: Securely authenticates with AWS using a named profile from your local AWS CLI configuration.

How It Works 🧠
    The tool uses a Retrieval-Augmented Generation (RAG) pipeline.

    Data Ingestion: The application reads text from your uploaded PDFs or specified web links.

    Chunking: The raw text is split into smaller, manageable chunks.

    Embedding: These chunks are converted into numerical vectors using Bedrock's embedding model.

    Vector Store: The vectors are stored in a vector database (FAISS) for efficient semantic search.

    Retrieval: When you ask a question, the agent searches the vector store to find the most relevant text chunks.

    Generation: The retrieved chunks, along with your question, are sent as context to Bedrock's large language model, which formulates the final answer.


Usage
    On the sidebar, upload one or more PDF files or paste URLs into the text area.

    Click the "Process Documents" button. Wait for the success message.

    In the main content area, type your question into the text box and press Enter.

    The model will retrieve relevant context and provide an answer. If the answer is not found in the documents, it will respond with "I don't know."