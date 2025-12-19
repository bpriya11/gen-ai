# langgraph_pr_agent.py
# This script runs the LangGraph agent that orchestrates the PR review.

import operator
import os
import uuid
import asyncio
import boto3
from typing import Annotated, Dict, Any, List, TypedDict

import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult

# --- 1. Define the Agent's Shared State ---
class AgentState(TypedDict):
    pr_id: int
    repo_owner: str
    repo_name: str
    messages: Annotated[List[BaseMessage], operator.add]
    pr_details: Dict[str, Any]
    pr_diff: str
    linter_report: str
    rag_context: str
    review_comment: str
    mcp_session: ClientSession

# --- 2. Define the LLM, Embeddings, and Tools ---

@st.cache_resource
def get_llm():
    session = boto3.Session(profile_name='my-new-profile')
    bedrock = session.client('bedrock-runtime', region_name='us-east-1')
    bedrock_runtime = session.client("bedrock-runtime")
    return Bedrock(
        model_id="mistral.mixtral-8x7b-instruct-v0:1",
        client=bedrock_runtime,
        streaming=True
    )
    

@st.cache_resource
def get_embeddings():
    session = boto3.Session(profile_name='my-new-profile')
    bedrock_runtime = session.client("bedrock-runtime")
    return BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        client=bedrock_runtime
    )

@st.cache_resource
def get_vectorstore():
    with open("contributing.md", "r") as f:
        docs = [Document(page_content=f.read())]
    embeddings = get_embeddings()
    print("Vectorstore created with contributing.md content.",embeddings)
    dd=FAISS.from_documents(docs, embeddings)
    print("Vectorstore234356.",dd)
    return dd

async def fetch_pr_details_tool(repo_owner: str, repo_name: str, pr_number: int, session: ClientSession) -> Dict[str, Any]:
    print(f"Session ID: {session}", flush=True)
    print(f"Fetching PR details forrrrrr {repo_owner}/{repo_name}#{pr_number}", flush=True)
    result: CallToolResult = await session.call_tool(
                    "fetch_pr_details",
                    arguments={
                        "repo_owner": repo_owner,
                        "repo_name": repo_name,
                        "pr_number": pr_number
                    }
                )
    print(f"PR details fetched: {result}", flush=True)
    return result.structuredContent.get('result', {})

async def fetch_pr_diff_tool(repo_owner: str, repo_name: str, pr_number: int, session: ClientSession) -> str:
    print(f"Fetching PR diff for {repo_owner}/{repo_name}#{pr_number}", flush=True)
    result: CallToolResult = await session.call_tool("fetch_pr_diff", {"repo_owner": repo_owner, "repo_name": repo_name, "pr_number": pr_number})
    return result.content[0].text

async def post_pr_comment_tool(repo_owner: str, repo_name: str, pr_number: int, comment: str, session: ClientSession) -> bool:
    print(f"Posting comment on {repo_owner}/{repo_name}#{pr_number}", flush=True)
    result: CallToolResult = await session.call_tool("post_pr_comment", {"repo_owner": repo_owner, "repo_name": repo_name, "pr_number": pr_number, "comment": comment})
    print(f"Comment posted: {result}", flush=True)
    return result.structuredContent.get("result", False)

async def run_mock_linter_tool(pr_diff: str, session: ClientSession) -> str:
    print(f"Running mock linter.", flush=True)
    result: CallToolResult = await session.call_tool("run_mock_linter", {"pr_diff": pr_diff})
    print(f"Mock linter result: {result}", flush=True)
    return result.content[0].text

# --- 3. Define the Nodes of the Graph ---

async def fetch_pr_info(state: AgentState):
    print("---NODE---: Fetching PR info.", flush=True)
    session = state['mcp_session']
    pr_details = await fetch_pr_details_tool(state['repo_owner'], state['repo_name'], state['pr_id'], session)
    pr_diff = await fetch_pr_diff_tool(state['repo_owner'], state['repo_name'], state['pr_id'], session)
    return {"pr_details": pr_details, "pr_diff": pr_diff}

async def run_linter(state: AgentState):
    print("---NODE---: Running linter.", flush=True)
    session = state['mcp_session']
    pr_diff = state['pr_diff']
    linter_report = await run_mock_linter_tool(pr_diff, session)
    return {"linter_report": linter_report}

def retrieve_context(state: AgentState):
    print("---NODE---: Retrieving RAG context.", flush=True)
    vectorstore = get_vectorstore()
    query = f"Project style guide and PR best practices based on: \nPR Title: {state['pr_details']['title']}\nPR Body: {state['pr_details']['body']}"
    docs = vectorstore.similarity_search(query, k=1)
    print("hii 13.", flush=True)
    context = docs[0].page_content if docs else "No specific guidelines found."
    return {"rag_context": context}

async def generate_review(state: AgentState):
    print("---NODE---: Generating review comment.", flush=True)
    
    llm = get_llm()
    print("hii 14.", llm, flush=True)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Senior Staff Engineer and a meticulous code reviewer. Your task is to provide a comprehensive, professional, and actionable review of a pull request."),
        ("human", "Here is all the context you need to perform the review. Synthesize it all into a single, comprehensive review comment. Always start with a summary of the changes.\n\n"
                  "### Pull Request Details:\n"
                  "PR Title: {title}\n"
                  "PR Body: {body}\n"
                  "\n### Code Changes:\n{diff}\n\n"
                  "### Linter Report:\n{linter_report}\n\n"
                  "### Project Guidelines:\n{rag_context}\n\n"
                  "Based on this information, perform the following:\n"
                  "1. **Summarize the changes** in a high-level overview.\n"
                  "2. **Identify and Explain Issues** that go beyond the linter. Look for:\n"
                  "    - Performance or security bottlenecks.\n"
                  "    - Architectural or design flaws.\n"
                  "    - Readability or maintainability concerns.\n"
                  "3. **Propose concrete solutions** or refactorings for the issues you've identified.\n"
                  "4. **Use a professional and constructive tone.**\n\n"
                  "Generate your final review comment now."),
    ])

    chain = prompt_template | llm | StrOutputParser()
    print("Generating review comment with chain:", flush=True)
    try:
        review_comment = await chain.ainvoke({
            "title": state['pr_details']['title'],
            "body": state['pr_details']['body'],
            "diff": state['pr_diff'],
            "linter_report": state['linter_report'],
            "rag_context": state['rag_context']
        })
    except Exception as e:
        print(f"Error during chain.ainvoke: {e}", flush=True)
        review_comment = "Error: " + str(e)

    print(f"Generated review comment: {review_comment}", flush=True)
    
    return {"review_comment": review_comment}

async def post_review_comment(state: AgentState):
    print("---NODE---: Posting review comment.", flush=True)
    session = state['mcp_session']
    comment = f"### Automated PR Review\n\n{state['review_comment']}"

    await post_pr_comment_tool(state['repo_owner'], state['repo_name'], state['pr_id'], comment, session)
    print("Review comment posted successfully.", flush=True)
    return END

# --- 4. Main application and Graph compilation ---

st.title("🤖 Automated Pull Request Reviewer")
st.info("This agent uses LangGraph and an MCP server to automatically review a GitHub PR.")

pr_url = st.text_input("Enter GitHub PR URL:", "https://github.com/bpriya11/test_priya/pull/1")
if 'status_message' not in st.session_state:
    st.session_state.status_message = "Ready."

st.divider()

async def run_review_workflow(pr_id: int, repo_owner: str, repo_name: str):
    print("Launching MCP server subprocess...", flush=True)
    env_vars =None
    
    server_params = StdioServerParameters(command="python", args=["mcp_git_server.py"], env=env_vars)
    
    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                st.session_state.status_message = "MCP server and client session are active."
                st.write(st.session_state.status_message)
                
                workflow = StateGraph(AgentState)
                workflow.add_node("fetch_info", fetch_pr_info)
                workflow.add_node("linter", run_linter)
                workflow.add_node("retrieve_context", retrieve_context)
                workflow.add_node("generate_review", generate_review)
                workflow.add_node("post_comment", post_review_comment)

                workflow.set_entry_point("fetch_info")
                workflow.add_edge("fetch_info", "linter")
                workflow.add_edge("linter", "retrieve_context")
                workflow.add_edge("retrieve_context", "generate_review")
                workflow.add_edge("generate_review", "post_comment")

                app = workflow.compile()
                
                initial_state = {
                    "pr_id": pr_id,
                    "repo_owner": repo_owner,
                    "repo_name": repo_name,
                    "messages": [HumanMessage(content="PR review request initiated.")],
                    "pr_details": {},
                    "pr_diff": "",
                    "linter_report": "",
                    "rag_context": "",
                    "review_comment": "",
                    "mcp_session": session
                }

                await app.ainvoke(initial_state)

        st.session_state.status_message = "Review process completed successfully."
        st.success("🎉 Review process completed! Check your repository for the comment.")

    except Exception as e:
        st.session_state.status_message = f"Error: {str(e)}"
        st.error(f"❌ An error occurred: {str(e)}")

if st.button("Run Review on PR"):
    st.session_state.status_message = "Starting review..."
    try:
        parts = pr_url.split('/')
        repo_owner = parts[3]
        repo_name = parts[4]
        pr_number = int(parts[6])
        
        asyncio.run(run_review_workflow(pr_number, repo_owner, repo_name))
    
    except (IndexError, ValueError) as e:
        st.session_state.status_message = f"Error: Invalid PR URL format. {str(e)}"
        st.error(f"Invalid PR URL format. Please enter a valid GitHub PR URL. Error: {e}")

st.write(f"Status: {st.session_state.status_message}")