import os
import sys
import subprocess


def check_api_key():
    """Check if API key is available."""
    api_key = os.environ.get("MXBAI_API_KEY")
    if not api_key:
        print("MXBAI_API_KEY environment variable is not set!")
        print("Please set your API key:")
        print("export MXBAI_API_KEY='your-api-key-here'")
        return False
    return True


def run_unit_tests():
    """Run unit tests only (no API calls)."""
    print("Running unit tests...")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    cmd = ["python", "-m", "pytest", "-m", "not integration", "tests/"]
    return subprocess.run(cmd, env=env).returncode


def run_integration_tests():
    """Run integration tests (requires API key)."""
    print("Running integration tests...")
    if not check_api_key():
        return 1

    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    cmd = ["python", "-m", "pytest", "-m", "integration", "tests/", "-v"]
    return subprocess.run(cmd, env=env).returncode


def run_specific_test(test_name):
    """Run a specific test."""
    print(f"Running specific test: {test_name}")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    cmd = ["python", "-m", "pytest", "-k", test_name, "tests/", "-v"]
    return subprocess.run(cmd, env=env).returncode


def run_all_tests():
    """Run all tests."""
    print("Running all tests...")
    if not check_api_key():
        print("API key not found. Integration tests will be skipped.")

    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    cmd = ["python", "-m", "pytest", "tests/", "-v"]
    return subprocess.run(cmd, env=env).returncode


def run_specific_module(module_name):
    """Run tests for a specific module."""
    print(f"Running tests for module: {module_name}")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    test_file = f"tests/test_{module_name}.py"
    if not os.path.exists(test_file):
        print(f"Test file {test_file} does not exist!")
        return 1
    cmd = ["python", "-m", "pytest", test_file, "-v"]
    return subprocess.run(cmd, env=env).returncode


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py [unit|integration|all|module_name|test_name]")
        print("")
        print("Commands:")
        print("  unit        - Run unit tests only (no API calls)")
        print("  integration - Run integration tests (requires API key)")
        print("  all         - Run all tests")
        print("  embeddings  - Run embeddings tests only")
        print("  reranker    - Run reranker tests only")
        print("  retrievers  - Run retriever tests only")
        print("  document_loader - Run document loader tests only")
        print("  <test_name> - Run specific test (partial name matching)")
        print("")
        print("Examples:")
        print("  python run_tests.py unit")
        print("  python run_tests.py integration")
        print("  python run_tests.py embeddings")
        print("  python run_tests.py test_integration_basic_embedding")
        return 1

    command = sys.argv[1].lower()

    if command == "unit":
        return run_unit_tests()
    elif command == "integration":
        return run_integration_tests()
    elif command == "all":
        return run_all_tests()
    elif command in ["embeddings", "reranker", "retrievers", "document_loader"]:
        return run_specific_module(command)
    else:
        return run_specific_test(command)


if __name__ == "__main__":
    sys.exit(main())
