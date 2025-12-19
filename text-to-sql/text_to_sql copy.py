import streamlit as st
import psycopg2
import csv
import boto3
import os
import json
import pandas as pd
import time
from dotenv import load_dotenv
from langchain.embeddings import BedrockEmbeddings
from langchain.llms import Bedrock
from opensearchpy import OpenSearch


load_dotenv()
st.set_page_config(page_title="Text to SQL Chat")
st.title("💬 Text to SQL Chat Interface")

# Sidebar DB connection inputs
st.sidebar.header("PostgreSQL DB Connection")
host = st.sidebar.text_input("Host",value="localhost")
port = st.sidebar.text_input("Port", value="5432")
db = st.sidebar.text_input("Database",value="rag")
user = st.sidebar.text_input("Username",value="postgres")
password = st.sidebar.text_input("Password", type="password")
submit = st.sidebar.button("Submit & Generate Embeddings")

opensearch_host = os.environ.get("OPEN_SEARCH_HOST")
opensearch_port = os.environ.get("OPEN_SEARCH_PORT")
opensearch_user = os.environ.get("OPEN_SEARCH_USER")
opensearch_password = os.environ.get("OPEN_SEARCH_PASSWORD")

# Initialize OpenSearch client globally
if opensearch_host and opensearch_port and opensearch_user and opensearch_password:
    os_client = OpenSearch(
        hosts=[{"host": opensearch_host, "port": opensearch_port}],
        http_auth=(opensearch_user, opensearch_password),
        use_ssl=True,
        verify_certs=True
    )

# Initialize shared DB connection
if "conn" not in st.session_state:
    st.session_state.conn = None

# Run initial schema extraction and embedding
if submit:
    try:
        st.info("🔌 Connecting to PostgreSQL DB...")
        st.session_state.conn = psycopg2.connect(host=host, port=port, dbname=db, user=user, password=password)
        cursor = st.session_state.conn.cursor()
        st.info("📋 Fetching table names...")
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema='public' AND table_type='BASE TABLE';
        """)
        tables = [row[0] for row in cursor.fetchall()]

        # Initialize schema structure with relationships
        schema = {"tables": [], "relationships": []}

        for table in tables:
            st.info(f"🔍 Extracting schema for table '{table}'...")
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s;
            """, (table,))
            columns = cursor.fetchall()
            schema["tables"].append({
                "name": table,
                "columns": [{"name": col[0], "type": col[1]} for col in columns]
            })

            # --- NEW: Extract Foreign Keys (Relationships) ---
            # This query gets all foreign keys where 'table' is either the foreign table or the referenced table
            cursor.execute(f"""
                SELECT
                    tc.constraint_name,
                    kcu.table_name AS local_table,
                    kcu.column_name AS local_column,
                    ccu.table_name AS foreign_table,
                    ccu.column_name AS foreign_column
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND (kcu.table_name = %s OR ccu.table_name = %s);
            """, (table, table)) # Pass table name twice for both conditions
            foreign_keys = cursor.fetchall()
            for fk in foreign_keys:
                # Add relationship if not already added (to avoid duplicates from both sides of the FK)
                rel_info = {
                    "constraint_name": fk[0],
                    "local_table": fk[1],
                    "local_column": fk[2],
                    "foreign_table": fk[3],
                    "foreign_column": fk[4]
                }
                if rel_info not in schema["relationships"]:
                    schema["relationships"].append(rel_info)

        cursor.close()

        schema_path = "/tmp/schema.json" # Local path for temp storage
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)

        
        st.success("✅ Schema extracted.")

        st.info("🧠 Generating schema embeddings using Amazon Bedrock...")
        embedder = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")

        schema_chunks = []
        doc_ids_for_os = [] # To store unique IDs for OpenSearch

        # Add table definitions to chunks
        for table_info in schema["tables"]:
            col_text = ", ".join([f"{col['name']} ({col['type']})" for col in table_info["columns"]])
            chunk = f"Table: {table_info['name']}. Columns: {col_text}."
            schema_chunks.append(chunk)
            doc_ids_for_os.append(f"{host}:{port}:{db}:{table_info['name']}")

        # Add relationship definitions to chunks
        for rel in schema["relationships"]:
            relationship_text = (
                f"Relationship: The '{rel['local_table']}' table's '{rel['local_column']}' column "
                f"references the '{rel['foreign_table']}' table's '{rel['foreign_column']}' column. "
                f"This means you can JOIN '{rel['local_table']}' and '{rel['foreign_table']}' "
                f"ON {rel['local_table']}.{rel['local_column']} = {rel['foreign_table']}.{rel['foreign_column']}."
            )
            schema_chunks.append(relationship_text)
            # Create a unique ID for the relationship embedding
            doc_ids_for_os.append(f"{host}:{port}:{db}:relationship:{rel['local_table']}_to_{rel['foreign_table']}_{rel['local_column']}")

        embeddings = embedder.embed_documents(schema_chunks)

        st.info("📦 Indexing embeddings into OpenSearch...")

        # Ensure the index exists, create if not
        if not os_client.indices.exists(index="schema-vectors"):
            index_body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.space_type": "cosinesimil"
                    }
                },
                "mappings": {
                    "properties": {
                        "vector": {
                            "type": "knn_vector",
                            "dimension": 1536 # Dimension for Titan Embeddings
                        },
                        "text": {
                            "type": "text"
                        },
                        "source_db": {
                            "type": "keyword"
                        }
                    }
                }
            }
            os_client.indices.create(index="schema-vectors", body=index_body)
            st.success("Created OpenSearch index 'schema-vectors'.")


        for i, (doc, vector, doc_id_os) in enumerate(zip(schema_chunks, embeddings, doc_ids_for_os)):
            os_client.index(index="schema-vectors", id=doc_id_os, body={
                "text": doc,
                "vector": vector,
                "source_db": doc_id_os # Store full doc_id
            })

        st.success("✅ Schema embeddings indexed into OpenSearch. Ready to generate SQL queries!")

    except Exception as e:
        st.error(f"❌ Error during setup: {str(e)}")
        if "password authentication failed" in str(e):
            st.error("Double-check your PostgreSQL username and password.")
        elif "connection refused" in str(e):
            st.error("Ensure your PostgreSQL database is running and accessible from this machine (check host/port and firewall).")


# === Chat-style session ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0")
embedder = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")


st.subheader("💬 Ask your question")
query_input = st.text_input("User question", key="chat_input")
if st.button("Submit Query") and query_input: # Ensure query_input is not empty
    st.session_state.chat_history.append(("user", query_input))

    if st.session_state.conn is None:
        st.error("Please connect to the database first using the sidebar inputs.")
        st.stop()

    try:
        query_embedding = embedder.embed_query(query_input)
        search_body = {
            "size": 5, # Increased size to get more relevant schema info
            "query": {
                "knn": {
                    "vector": {
                        "vector": query_embedding,
                        "k": 5 # Increased k for more context
                    }
                }
            },
            "_source": ["text"] # Only retrieve the text field
        }
        search_results = os_client.search(index="schema-vectors", body=search_body)
        context = "\n".join([hit["_source"]["text"] for hit in search_results["hits"]["hits"]])


        base_prompt = f"""
You are an expert SQL query generator. Your task is to translate natural language questions into **syntactically correct and efficient PostgreSQL queries**.

You MUST strictly follow all the rules below.

### 📜 Database Schema and Relationships:
{context}
---

### 🔐 STRICT SQL RULES:
- ❌ **DO NOT use `SELECT *`.** Always list explicit column names.
- ✅ **Table Aliases:** When using `JOIN` statements, always use short, meaningful aliases for tables (e.g., `p` for `products`, `oi` for `order_items`, `o` for `orders`, `c` for `customers`).
- ✅ **Joining Tables:** Only `JOIN` tables when necessary to fulfill the query. Use the foreign key relationships provided in the schema context for `ON` clauses.
- ✅ **`GROUP BY` Rule:** If you use any aggregate function (`SUM`, `COUNT`, `AVG`, `MIN`, `MAX`) in your `SELECT` clause, **ALL other non-aggregated columns in your `SELECT` clause MUST also appear in your `GROUP BY` clause.**
- ❌ **`SELECT *` with `GROUP BY` is FORBIDDEN.**
- ✅ **Aggregates with Aliases:** Always use aggregates with clear aliases (e.g., `SUM(oi.quantity) AS total_quantity`).
- ✅ **Order of Clauses:** `ORDER BY` must always come before `LIMIT`.
- ✅ **Output Format:** Always format valid PostgreSQL queries. Do not return explanations, comments, or any text other than the SQL query itself.

Schema:
{context}

User question:
{query_input}

SQL:
"""

        sql_query = llm.predict(base_prompt).strip() # Initial generation

        attempt = 0
        max_attempts = 4
        success = False

        while attempt < max_attempts and not success:
            st.markdown(f"**Generated SQL (Attempt {attempt + 1}):**\n```sql\n{sql_query}\n```")

            # --- IMPORTANT: Robust SQL Extraction ---
            # The LLM might output markdown. Extract just the SQL.
            # Handle cases where LLM might put extra text before/after the code block
            if "```sql" in sql_query:
                try:
                    sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
                except IndexError: # In case the closing ``` is missing
                    st.warning("LLM output malformed, trying simpler strip.")
                    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            elif "```" in sql_query: # Fallback if it uses generic code block
                try:
                    sql_query = sql_query.split("```")[1].split("```")[0].strip()
                except IndexError:
                    st.warning("LLM output malformed, trying simpler strip.")
                    sql_query = sql_query.replace("```", "").strip()
            # If no markdown block, assume the whole output is the query.
            
            # Further simple cleanup just in case LLM adds prefixes like "SQL:"
            if sql_query.lower().startswith("sql:"):
                sql_query = sql_query[len("sql:"):].strip()
            # Also remove any trailing semicolons if not part of query structure
            if sql_query.endswith(';'):
                sql_query = sql_query[:-1]


            try:
                cursor = st.session_state.conn.cursor()
                cursor.execute(sql_query)
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                cursor.close()
                st.session_state.chat_history.append(("sql", sql_query))
                st.session_state.chat_history.append(("result", pd.DataFrame(rows, columns=columns)))
                success = True
                break # Exit loop on success
            except Exception as e:
                attempt += 1
                error_msg = str(e)
                st.session_state.conn.rollback() # Rollback in case of error

                st.error(f"❌ Error: {error_msg}")
                if attempt < max_attempts: # Only show retry message if more attempts are left
                    st.info(f"🔄 Attempt {attempt + 1}/{max_attempts} to correct SQL...")

                    correction_prompt = f"""
                    The previous SQL query caused an error in PostgreSQL. Please analyze the error message and correct the query.

                    Original User Question:
                    {query_input}

                    Previous Invalid SQL Query:
                    ```sql
                    {sql_query}
                    ```

                    PostgreSQL Error Message:
                    {error_msg}

                    ### 🔐 STRICT SQL RULES (Reiterate and Focus on common errors):
                    - ❌ **DO NOT use `SELECT *`.** List explicit column names.
                    - ✅ **CRITICAL: If using `GROUP BY`, ensure all non-aggregated columns in your `SELECT` clause are also included in your `GROUP BY` clause.** This is a frequent cause of the error: "column must appear in the GROUP BY clause or be used in an aggregate function".
                    - ✅ **JOINs:** Use appropriate JOINs based on the schema and explicit foreign key relationships provided.
                    - ✅ **Output Format:** ONLY return the corrected SQL query. Do NOT include any explanation, comments, or extra text.

                    Corrected SQL:
                    """
                    sql_query = llm.predict(correction_prompt).strip()
                else:
                    # If no more attempts, log the final failed query
                    st.session_state.chat_history.append(("error", f"Failed after {max_attempts} attempts. Last query:\n```sql\n{sql_query}\n```\nError: {error_msg}"))
                    break # Exit loop as no more attempts

    except Exception as e:
        st.session_state.chat_history.append(("error", f"An unexpected error occurred: {str(e)}"))

# Display chat history
for speaker, content in st.session_state.chat_history:
    if speaker == "user":
        st.markdown(f"**You:** {content}")
    elif speaker == "sql":
        st.markdown(f"**SQL:**\n```sql\n{content}\n```")
    elif speaker == "result":
        st.dataframe(content)
    elif speaker == "error":
        st.error(content)
