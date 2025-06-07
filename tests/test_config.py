import os
from typing import Optional, List


class TestConfig:
    """Configuration class for integration tests."""

    API_KEY = os.environ.get("MXBAI_API_KEY")

    OFFICIAL_API_BASE_URL = "https://api.mixedbread.com"
    CUSTOM_BASE_URL = os.environ.get("MXBAI_CUSTOM_BASE_URL")

    # Test vector store IDs for retriever integration tests
    TEST_VECTOR_STORE_IDS = os.environ.get("MXBAI_TEST_VECTOR_STORE_IDS")

    TEST_TIMEOUT = 30.0
    TEST_MAX_RETRIES = 2

    DEFAULT_EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
    DEFAULT_RERANKING_MODEL = "mixedbread-ai/mxbai-rerank-large-v2"

    # Test file paths for document parsing tests
    TEST_PDF_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "acme_invoice.pdf"
    )

    @classmethod
    def get_base_url(cls) -> Optional[str]:
        """Get the base URL to use for tests."""
        return cls.CUSTOM_BASE_URL or cls.OFFICIAL_API_BASE_URL

    @classmethod
    def has_api_key(cls) -> bool:
        """Check if API key is available."""
        return bool(cls.API_KEY)

    @classmethod
    def get_test_vector_store_ids(cls) -> Optional[List[str]]:
        """Get test vector store IDs for retriever tests."""
        if cls.TEST_VECTOR_STORE_IDS:
            return [
                id.strip() for id in cls.TEST_VECTOR_STORE_IDS.split(",") if id.strip()
            ]
        return None

    @classmethod
    def get_test_embedder_config(cls) -> dict:
        """Get default configuration for embedder tests."""
        config = {
            "timeout": cls.TEST_TIMEOUT,
            "max_retries": cls.TEST_MAX_RETRIES,
            "model": cls.DEFAULT_EMBEDDING_MODEL,
        }

        if cls.API_KEY:
            config["api_key"] = cls.API_KEY

        base_url = cls.get_base_url()
        if base_url:
            config["base_url"] = base_url

        # Add test vector store IDs for retriever tests
        test_vector_store_ids = cls.get_test_vector_store_ids()
        if test_vector_store_ids:
            config["test_vector_store_ids"] = test_vector_store_ids

        return config

    @classmethod
    def get_test_reranker_config(cls) -> dict:
        """Get default configuration for reranker tests."""
        config = {
            "timeout": cls.TEST_TIMEOUT,
            "max_retries": cls.TEST_MAX_RETRIES,
            "model": cls.DEFAULT_RERANKING_MODEL,
        }

        if cls.API_KEY:
            config["api_key"] = cls.API_KEY

        base_url = cls.get_base_url()
        if base_url:
            config["base_url"] = base_url

        return config

    @classmethod
    def get_test_document_loader_config(cls) -> dict:
        """Get default configuration for document loader tests."""
        config = {
            "timeout": cls.TEST_TIMEOUT,
            "max_retries": cls.TEST_MAX_RETRIES,
            "chunking_strategy": "page",
            "return_format": "markdown",
            "element_types": ["text", "title"],
            "max_wait_time": 300,
            "poll_interval": 5,
        }

        if cls.API_KEY:
            config["api_key"] = cls.API_KEY

        base_url = cls.get_base_url()
        if base_url:
            config["base_url"] = base_url

        return config

    @classmethod
    def get_test_retriever_config(cls) -> dict:
        """Get default configuration for retriever tests."""
        config = {
            "timeout": cls.TEST_TIMEOUT,
            "max_retries": cls.TEST_MAX_RETRIES,
            "top_k": 5,
        }

        if cls.API_KEY:
            config["api_key"] = cls.API_KEY

        base_url = cls.get_base_url()
        if base_url:
            config["base_url"] = base_url

        test_vector_store_ids = cls.get_test_vector_store_ids()
        if test_vector_store_ids:
            config["vector_store_ids"] = test_vector_store_ids

        return config
