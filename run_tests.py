#!/usr/bin/env python3
"""
Test runner script for mixedbread-ai-langchain

Usage:
    python run_tests.py all                    # Run all tests
    python run_tests.py unit                   # Unit tests only (no API key required)
    python run_tests.py integration            # Integration tests (requires MXBAI_API_KEY)
    python run_tests.py embeddings             # Component-specific tests
    python run_tests.py reranker               # Component-specific tests
    python run_tests.py retrievers             # Component-specific tests
    python run_tests.py document_loader        # Component-specific tests
    python run_tests.py <test_name>            # Specific test (partial name matching)
"""

import sys
import subprocess
import os


def run_pytest(args):
    """Run pytest with given arguments"""
    cmd = ["python", "-m", "pytest"] + args
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    test_type = sys.argv[1].lower()
    
    # Base pytest arguments
    base_args = ["-v", "--tb=short"]
    
    if test_type == "all":
        # Run all tests
        exit_code = run_pytest(base_args)
    
    elif test_type == "unit":
        # Run unit tests only (exclude integration)
        exit_code = run_pytest(base_args + ["-m", "not integration"])
    
    elif test_type == "integration":
        # Run integration tests only
        if not os.getenv("MXBAI_API_KEY"):
            print("ERROR: MXBAI_API_KEY environment variable is required for integration tests")
            sys.exit(1)
        exit_code = run_pytest(base_args + ["-m", "integration"])
    
    elif test_type in ["embeddings", "embedding"]:
        # Run embedding-related tests
        exit_code = run_pytest(base_args + ["-k", "embedding"])
    
    elif test_type in ["reranker", "rerank"]:
        # Run reranker-related tests
        exit_code = run_pytest(base_args + ["-k", "rerank"])
    
    elif test_type in ["retrievers", "retriever"]:
        # Run retriever-related tests
        exit_code = run_pytest(base_args + ["-k", "retriever"])
    
    elif test_type in ["document_loader", "loader"]:
        # Run document loader tests
        exit_code = run_pytest(base_args + ["-k", "document_loader"])
    
    else:
        # Treat as partial test name matching
        exit_code = run_pytest(base_args + ["-k", test_type])
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()