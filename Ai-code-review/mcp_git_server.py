# mcp_git_server.py
# This is a standalone MCP server that exposes tools to interact with Git.

import requests
import os
from dotenv import load_dotenv
import json
from typing import Dict, Any, List
import subprocess

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Create a FastMCP server instance
mcp = FastMCP("Git Server")
load_dotenv()

# Get a generic Git token from environment variables
GIT_TOKEN = os.environ.get("GIT_TOKEN")
if not GIT_TOKEN:
    print("FATAL ERROR: GIT_TOKEN environment variable not set.", flush=True)
    exit(1)

# Use GitHub API URL and headers
GITHUB_API_URL = "https://api.github.com"
GITHUB_HEADERS_DIFF = {
    "Authorization": f"token {GIT_TOKEN}",
    "Accept": "application/vnd.github.v3.diff",
}
GITHUB_HEADERS_JSON = {
    "Authorization": f"Bearer {GIT_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

@mcp.tool()
def fetch_pr_details(repo_owner: str, repo_name: str, pr_number: int) -> Dict[str, Any]:
    """Fetches details (title, body, url) of a pull request."""
    print(f"MCP Server: Fetching PR details for {repo_owner}/{repo_name}#{pr_number}", flush=True)
    try:
        url = f"{GITHUB_API_URL}/repos/{repo_owner}/{repo_name}/pulls/{pr_number}"
        response = requests.get(url, headers=GITHUB_HEADERS_JSON, verify=False) # 
        # --- NEW DEBUGGING PRINTS ---
        
        # response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        pr_data = response.json()
        print("MCP Server: Successfully fetched PR details.", flush=True)
        return {
            "title": pr_data.get("title"),
            "body": pr_data.get("body", ""),
            "url": pr_data.get("html_url"),
            "head_sha": pr_data.get("head", {}).get("sha")
        }
    except requests.exceptions.RequestException as e:
        print(f"MCP Server ERROR: Failed to fetch PR details. Reason: {e}", flush=True)
        return {"error": str(e), "status_code": response.status_code if 'response' in locals() else None}
    except json.JSONDecodeError as e:
        print(f"MCP Server ERROR: Malformed JSON response. Reason: {e}", flush=True)
        print(f"MCP Server DEBUG: Response content was: {response.text}", flush=True)
        return {"error": f"JSON decode error: {e}"}

@mcp.tool()
def fetch_pr_diff(repo_owner: str, repo_name: str, pr_number: int) -> str:
    """Fetches the code diff of a pull request as a string."""
    print(f"MCP Server: Fetching PR diff for {repo_owner}/{repo_name}#{pr_number}", flush=True)
    try:
        url = f"{GITHUB_API_URL}/repos/{repo_owner}/{repo_name}/pulls/{pr_number}"
        response = requests.get(url, headers=GITHUB_HEADERS_DIFF, verify=False) # <--- ADDED 'verify=False' for debugging
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to fetch PR diff. Reason: {e}"

@mcp.tool()
def post_pr_comment(repo_owner: str, repo_name: str, pr_number: int, comment: str) -> bool:
    """Posts a comment on a pull request."""
    print(f"MCP Server: Posting comment on {repo_owner}/{repo_name}#{pr_number}", flush=True)
    try:
        url = f"{GITHUB_API_URL}/repos/{repo_owner}/{repo_name}/issues/{pr_number}/comments"
        data = {"body": comment}
        response = requests.post(url, json=data, headers=GITHUB_HEADERS_JSON, verify=False) # <--- ADDED 'verify=False' for debugging
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        return False

@mcp.tool()
def run_linter(pr_diff: str) -> str:
    """
    Runs a real linter (flake8) on the code diff.
    """
    print("MCP Server: Running flake8 linter on PR diff.", flush=True)

    try:
        # We're passing the diff via stdin, which is a robust way to handle code snippets.
        result = subprocess.run(
            ['flake8', '-'], # The '-' tells flake8 to read from stdin
            input=pr_diff.encode('utf-8'),
            capture_output=True,
            text=True,
            timeout=10 # Set a timeout for the linter process
        )

        # Check if flake8 returned any errors
        if result.returncode != 0:
            return f"Linter Report: {result.stdout.strip()}"
        else:
            return "Linter Report: No issues found."

    except FileNotFoundError:
        return "Error: Flake8 command not found. Please install Flake8 in the environment."
    except Exception as e:
        return f"Error: An unexpected error occurred while running the linter: {str(e)}"

def main():
    """Main entry point for the MCP server."""
    print("Starting MCP Git server...", flush=True)
    try:
        mcp.run()
    except BaseException as e:
        print(f"FATAL ERROR: MCP Server crashed. Reason: {e}", flush=True)

if __name__ == "__main__":
    main()