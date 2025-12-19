# Automated Pull Request Reviewer and Summarizer

This project is a complete end-to-end solution for an AI agent that automatically reviews and summarizes GitHub pull requests. It demonstrates a modern, modular architecture using:

- **LangGraph:** The agent's "brain," orchestrating a multi-step workflow.
- **Model Context Protocol (MCP):** A standardized way to interact with external services (in this case, the GitHub API) via a dedicated microservice.
- **RAG (Retrieval Augmented Generation):** The agent uses a small, project-specific knowledge base (our style guide) to ground its review.

The agent's workflow is triggered by a user input and follows a linear chain:
1.  **Fetch PR Info:** Connects to the MCP server to get the PR's title, body, and code diff.
2.  **Run Linter:** Calls a tool on the MCP server to run a (mock) linter.
3.  **Retrieve Context:** Performs a RAG search on the project's style guide to provide context for the LLM.
4.  **Generate Review:** Uses an LLM to synthesize all the information and draft a professional review comment.
5.  **Post Comment:** Calls an MCP tool to post the final comment on the PR.

---

### Prerequisites

You need to have Python 3.9+ installed.

1.  **GitHub Personal Access Token:** To allow the MCP server to read PRs and post comments.
    * Go to your GitHub profile settings -> Developer settings -> Personal access tokens -> Tokens (classic).
    * Click "Generate new token."
    * Give it a descriptive name (e.g., "PR Reviewer Agent").
    * Select the `repo` scope to grant full control of private repositories and `public_repo` for public ones.
    * Click "Generate token" and copy the token immediately.

### Setup and Installation

1.  **Clone this repository** (or create the files in a new directory).
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set environment variables:**
    ```bash
    export GITHUB_TOKEN="your-github-personal-access-token"
    ```

### How to Run the Solution

1.  **Start the agent:**
    ```bash
    streamlit run langgraph_pr_agent.py
    ```
2.  **Use the interface:**
    * Open your web browser and navigate to the Streamlit app.
    * Enter the URL of a GitHub pull request you want to test.
    * Click "Run Review on PR."
