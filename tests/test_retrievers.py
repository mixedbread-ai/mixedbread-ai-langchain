import pytest
import asyncio
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from mixedbread_ai_langchain import (
    MixedbreadVectorStoreRetriever,
    MixedbreadVectorStoreFileRetriever,
    MixedbreadVectorStoreManager,
)
from .test_config import TestConfig


class TestMixedbreadVectorStoreRetriever:
    def test_init_default(self, monkeypatch):
        """
        Test default initialization parameters for MixedbreadVectorStoreRetriever.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        retriever = MixedbreadVectorStoreRetriever(
            vector_store_ids=["test-store-1", "test-store-2"]
        )

        assert retriever.vector_store_ids == ["test-store-1", "test-store-2"]
        assert retriever.top_k == 10
        assert retriever.return_metadata is True

    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for MixedbreadVectorStoreRetriever.
        """
        retriever = MixedbreadVectorStoreRetriever(
            vector_store_ids=["test-store"],
            api_key="test-api-key",
            top_k=5,
            score_threshold=0.8,
            return_metadata=False,
        )

        assert retriever.vector_store_ids == ["test-store"]
        assert retriever.top_k == 5
        assert retriever.score_threshold == 0.8
        assert retriever.return_metadata is False

    def test_init_fail_wo_api_key(self, monkeypatch):
        """
        Test that initialization fails when no API key is provided.
        """
        monkeypatch.delenv("MXBAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Mixedbread API key not found"):
            MixedbreadVectorStoreRetriever(vector_store_ids=["test-store"])

    def test_init_fail_wo_vector_store_ids(self):
        """
        Test that initialization fails when no vector store IDs are provided.
        """
        with pytest.raises(
            ValueError, match="At least one vector_store_id must be provided"
        ):
            MixedbreadVectorStoreRetriever(vector_store_ids=[], api_key="fake-api-key")

    @patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.MixedbreadClient")
    def test_get_relevant_documents_empty_response(self, mock_client_class):
        """
        Test _get_relevant_documents with empty response.
        """
        # Mock the client and response
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.client.vector_stores.search.return_value = Mock(data=[])

        retriever = MixedbreadVectorStoreRetriever(
            vector_store_ids=["test-store"], api_key="fake-api-key"
        )

        documents = retriever._get_relevant_documents("test query")
        assert documents == []
        
        # Verify the search was called with correct parameters
        mock_client.client.vector_stores.search.assert_called_once()
        call_args = mock_client.client.vector_stores.search.call_args[1]
        assert call_args["query"] == "test query"
        assert call_args["vector_store_ids"] == ["test-store"]
        assert call_args["top_k"] == 10

    @patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.MixedbreadClient")
    def test_get_relevant_documents_with_results(self, mock_client_class):
        """
        Test _get_relevant_documents with mock search results.
        """
        # Mock the client and response
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock search result data
        mock_result = Mock()
        mock_result.position = 0
        mock_result.content = "Test chunk content"
        mock_result.score = 0.95
        mock_result.file_id = "file1"
        mock_result.filename = "test.pdf"
        mock_result.vector_store_id = "test-store"
        mock_result.value = None
        mock_result.metadata = {"author": "test"}

        mock_client.client.vector_stores.search.return_value = Mock(data=[mock_result])

        retriever = MixedbreadVectorStoreRetriever(
            vector_store_ids=["test-store"],
            api_key="fake-api-key",
        )

        documents = retriever._get_relevant_documents("test query")
        assert len(documents) == 1
        assert isinstance(documents[0], Document)
        assert documents[0].page_content == "Test chunk content"
        assert documents[0].metadata["file_id"] == "file1"
        assert documents[0].metadata["filename"] == "test.pdf"
        assert documents[0].metadata["relevance_score"] == 0.95
        
        # Verify the search was called with correct parameters
        mock_client.client.vector_stores.search.assert_called_once()
        call_args = mock_client.client.vector_stores.search.call_args[1]
        assert call_args["query"] == "test query"
        assert call_args["vector_store_ids"] == ["test-store"]
        assert call_args["top_k"] == 10
        assert call_args["return_metadata"] is True
        
    @patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.AsyncMixedbread")
    def test_search_async_convenience_method(self, mock_async_client_class):
        """
        Test the search_async convenience method.
        """
        # Mock the async client
        mock_async_client = Mock()
        mock_async_client_class.return_value = mock_async_client
        
        retriever = MixedbreadVectorStoreRetriever(
            vector_store_ids=["test-store"],
            api_key="fake-api-key"
        )
        
        # Test that the convenience method returns the right awaitable
        result = retriever.search_async("test query")
        
        # Verify the async client search method was prepared with correct parameters
        mock_async_client.vector_stores.search.assert_called_once_with(
            query="test query",
            vector_store_ids=["test-store"],
            top_k=10,
            return_metadata=True
        )
        
    @patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.AsyncMixedbread")
    def test_direct_async_client_access(self, mock_async_client_class):
        """
        Test direct access to the async client.
        """
        # Mock the async client
        mock_async_client = Mock()
        mock_async_client_class.return_value = mock_async_client
        
        retriever = MixedbreadVectorStoreRetriever(
            vector_store_ids=["test-store"],
            api_key="fake-api-key"
        )
        
        # Test direct client access
        assert retriever.aclient == mock_async_client
        
        # Users can call retriever.aclient.vector_stores.search() directly
        # This gives them full access to the SDK without wrapper limitations

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_search(self):
        """
        Test basic search with real API call.
        Note: This test requires valid vector store IDs with documents.
        """
        # Skip this test if no vector store IDs are provided
        retriever_config = TestConfig.get_test_embedder_config()
        test_vector_store_ids = retriever_config.get("test_vector_store_ids")
        if not test_vector_store_ids:
            pytest.skip("No test vector store IDs provided")

        retriever = MixedbreadVectorStoreRetriever(
            vector_store_ids=test_vector_store_ids, top_k=3, **retriever_config
        )

        documents = retriever.get_relevant_documents("test query")

        # Basic assertions that work regardless of content
        assert isinstance(documents, list)
        for doc in documents:
            assert isinstance(doc, Document)
            assert hasattr(doc, "page_content")
            assert hasattr(doc, "metadata")
            assert isinstance(doc.metadata, dict)


class TestMixedbreadVectorStoreFileRetriever:
    def test_init_default(self, monkeypatch):
        """
        Test default initialization parameters for MixedbreadVectorStoreFileRetriever.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        retriever = MixedbreadVectorStoreFileRetriever(
            vector_store_ids=["test-store-1", "test-store-2"]
        )

        assert retriever.vector_store_ids == ["test-store-1", "test-store-2"]
        assert retriever.top_k == 10
        assert retriever.return_metadata is True

    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for MixedbreadVectorStoreFileRetriever.
        """
        retriever = MixedbreadVectorStoreFileRetriever(
            vector_store_ids=["test-store"],
            api_key="test-api-key",
            top_k=5,
            score_threshold=0.8,
            return_metadata=False,
        )

        assert retriever.vector_store_ids == ["test-store"]
        assert retriever.top_k == 5
        assert retriever.score_threshold == 0.8
        assert retriever.return_metadata is False

    @patch(
        "mixedbread_ai_langchain.retrievers.vector_store_file_retriever.MixedbreadClient"
    )
    def test_get_relevant_documents_empty_response(self, mock_client_class):
        """
        Test _get_relevant_documents with empty response.
        """
        # Mock the client and response
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.client.vector_stores.files.search.return_value = Mock(data=[])

        retriever = MixedbreadVectorStoreFileRetriever(
            vector_store_ids=["test-store"], api_key="fake-api-key"
        )

        documents = retriever._get_relevant_documents("test query")
        assert documents == []
        
        # Verify the search was called with correct parameters
        mock_client.client.vector_stores.files.search.assert_called_once()
        call_args = mock_client.client.vector_stores.files.search.call_args[1]
        assert call_args["query"] == "test query"
        assert call_args["vector_store_ids"] == ["test-store"]
        assert call_args["top_k"] == 10

    @patch(
        "mixedbread_ai_langchain.retrievers.vector_store_file_retriever.MixedbreadClient"
    )
    def test_get_relevant_documents_with_results(self, mock_client_class):
        """
        Test _get_relevant_documents with mock file search results.
        """
        # Mock the client and response
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock file search result data
        mock_result = Mock()
        mock_result.id = "file1"
        mock_result.filename = "test.pdf"
        mock_result.vector_store_id = "test-store"
        mock_result.score = 0.95
        mock_result.status = "completed"
        mock_result.usage_bytes = 1024
        mock_result.created_at = "2024-01-01T00:00:00Z"
        mock_result.version = "1.0"
        mock_result.metadata = {"author": "test"}

        # Mock chunk data within file result
        mock_chunk = Mock()
        mock_chunk.position = 0
        mock_chunk.content = "Test file content"
        mock_chunk.score = 0.95
        mock_chunk.file_id = "file1"
        mock_result.chunks = [mock_chunk]

        mock_client.client.vector_stores.files.search.return_value = Mock(
            data=[mock_result]
        )

        retriever = MixedbreadVectorStoreFileRetriever(
            vector_store_ids=["test-store"],
            api_key="fake-api-key",
        )

        documents = retriever._get_relevant_documents("test query")
        assert len(documents) == 1
        assert isinstance(documents[0], Document)
        assert documents[0].page_content == "Test file content"
        assert documents[0].metadata["file_id"] == "file1"
        assert documents[0].metadata["filename"] == "test.pdf"
        assert documents[0].metadata["relevance_score"] == 0.95
        
        # Verify the search was called with correct parameters
        mock_client.client.vector_stores.files.search.assert_called_once()
        call_args = mock_client.client.vector_stores.files.search.call_args[1]
        assert call_args["query"] == "test query"
        assert call_args["vector_store_ids"] == ["test-store"]
        assert call_args["top_k"] == 10
        assert call_args["return_metadata"] is True
        assert call_args["return_chunks"] is False
        
    @patch("mixedbread_ai_langchain.retrievers.vector_store_file_retriever.AsyncMixedbread")
    def test_file_search_async_convenience_method(self, mock_async_client_class):
        """
        Test the search_async convenience method for file retriever.
        """
        # Mock the async client
        mock_async_client = Mock()
        mock_async_client_class.return_value = mock_async_client
        
        retriever = MixedbreadVectorStoreFileRetriever(
            vector_store_ids=["test-store"],
            api_key="fake-api-key",
            include_chunks=True,
            chunk_limit=3
        )
        
        # Test that the convenience method returns the right awaitable
        result = retriever.search_async("test query")
        
        # Verify the async client search method was prepared with correct parameters
        mock_async_client.vector_stores.files.search.assert_called_once_with(
            query="test query",
            vector_store_ids=["test-store"],
            top_k=10,
            return_metadata=True,
            return_chunks=True,
            chunk_limit=3
        )
        
    @patch("mixedbread_ai_langchain.retrievers.vector_store_file_retriever.AsyncMixedbread")
    def test_file_direct_async_client_access(self, mock_async_client_class):
        """
        Test direct access to the async client for file retriever.
        """
        # Mock the async client
        mock_async_client = Mock()
        mock_async_client_class.return_value = mock_async_client
        
        retriever = MixedbreadVectorStoreFileRetriever(
            vector_store_ids=["test-store"],
            api_key="fake-api-key"
        )
        
        # Test direct client access
        assert retriever.aclient == mock_async_client
        
        # Users can call retriever.aclient.vector_stores.files.search() directly
        # This gives them full access to the SDK without wrapper limitations

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_file_search(self):
        """
        Test basic file search with real API call.
        Note: This test requires valid vector store IDs with documents.
        """
        # Skip this test if no vector store IDs are provided
        retriever_config = TestConfig.get_test_embedder_config()
        test_vector_store_ids = retriever_config.get("test_vector_store_ids")
        if not test_vector_store_ids:
            pytest.skip("No test vector store IDs provided")

        retriever = MixedbreadVectorStoreFileRetriever(
            vector_store_ids=test_vector_store_ids, top_k=3, **retriever_config
        )

        documents = retriever.get_relevant_documents("test query")

        # Basic assertions that work regardless of content
        assert isinstance(documents, list)
        for doc in documents:
            assert isinstance(doc, Document)
            assert hasattr(doc, "page_content")
            assert hasattr(doc, "metadata")
            assert isinstance(doc.metadata, dict)
