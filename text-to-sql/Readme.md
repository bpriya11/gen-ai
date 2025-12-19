📝 Text-to-SQL with RAG (AWS Bedrock + PostgreSQL + Streamlit)
This project converts natural language questions into SQL queries using a Retrieval-Augmented Generation (RAG) pipeline with AWS Bedrock (Mixtral 8x7B + Titan Embeddings).
It retrieves database schema context to reduce hallucinations, executes the SQL on PostgreSQL, and displays results in a Streamlit UI.

🚀 Features
Natural language → SQL translation using Mixtral 8x7B

Schema-aware SQL generation via RAG with Titan embeddings

PostgreSQL backend with query execution

Interactive Streamlit UI

Configurations stored securely in .env


🖥️ Usage
Open the Streamlit UI (http://localhost:8501)

Enter a question in plain English, e.g.:

"Show me the total sales by region for the last quarter."

The app retrieves schema context → generates SQL → executes → displays results.