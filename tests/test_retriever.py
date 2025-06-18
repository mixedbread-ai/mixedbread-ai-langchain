"""Test suite for MixedbreadVectorStoreRetriever."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain_core.documents import Document
from pydantic import SecretStr

from mixedbread_ai_langchain.retrievers.vector_store_retriever import MixedbreadVectorStoreRetriever


class MockSearchResponse:
    """Mock search response."""
    def __init__(self, data):
        self.data = data


class MockChunkResult:
    """Mock chunk search result."""
    def __init__(self, content="Test content", filename="test.pdf", score=0.8, chunk_type="text"):
        # New API structure uses text, ocr_text, transcription instead of content
        self.type = chunk_type
        if chunk_type == "text":
            self.text = content
        elif chunk_type == "image_url":
            self.ocr_text = content
        elif chunk_type == "audio_url":
            self.transcription = content
        else:
            self.text = content  # fallback
        
        self.filename = filename
        self.score = score
        self.chunk_index = 0
        self.file_id = "test_file_id"
        self.mime_type = "text/plain"
        self.metadata = {}


class MockFileResult:
    """Mock file search result."""
    def __init__(self, filename="test.pdf", score=0.8, file_id="file_123", chunks=None):
        self.filename = filename
        self.score = score
        self.id = file_id
        self.chunks = chunks or []
        self.vector_store_id = "test_vector_store"
        self.status = "completed"
        self.created_at = "2024-01-01T00:00:00Z"
        self.usage_bytes = 1024
        self.metadata = {}


class TestMixedbreadVectorStoreRetriever:
    """Test MixedbreadVectorStoreRetriever."""

    def test_init_basic(self):
        """Test basic initialization."""
        with patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.get_from_dict_or_env") as mock_env:
            mock_env.return_value = "test-api-key"
            
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store1", "store2"]
            )
            
            assert retriever.vector_store_identifiers == ["store1", "store2"]
            assert retriever.top_k == 10
            assert retriever.search_type == "chunk"
            assert retriever.score_threshold is None

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        with patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.get_from_dict_or_env") as mock_env:
            mock_env.return_value = "test-api-key"
            
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store1"],
                api_key="custom-key",
                top_k=5,
                search_type="file",
                score_threshold=0.7,
            )
            
            assert retriever.vector_store_identifiers == ["store1"]
            assert retriever.top_k == 5
            assert retriever.search_type == "file"
            assert retriever.score_threshold == 0.7

    def test_init_validation_errors(self):
        """Test initialization validation errors."""
        with patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.get_from_dict_or_env") as mock_env:
            mock_env.return_value = "test-api-key"
            
            # Empty vector store identifiers
            with pytest.raises(ValueError, match="At least one vector_store_identifier must be provided"):
                MixedbreadVectorStoreRetriever(vector_store_identifiers=[])
            
            # Invalid search type
            with pytest.raises(ValueError, match="search_type must be 'chunk' or 'file'"):
                MixedbreadVectorStoreRetriever(
                    vector_store_identifiers=["store1"],
                    search_type="invalid"
                )

    def test_convert_chunk_results(self):
        """Test conversion of chunk search results."""
        with patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.get_from_dict_or_env") as mock_env:
            mock_env.return_value = "test-api-key"
            
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store1"],
                search_type="chunk"
            )
            
            mock_response = MockSearchResponse([
                MockChunkResult("Content 1", "file1.pdf", 0.9),
                MockChunkResult("Content 2", "file2.pdf", 0.7),
            ])
            
            documents = retriever._convert_results_to_documents(mock_response)
            
            assert len(documents) == 2
            assert documents[0].page_content == "Content 1"
            assert documents[0].metadata["filename"] == "file1.pdf"
            assert documents[0].metadata["score"] == 0.9
            
            assert documents[1].page_content == "Content 2"
            assert documents[1].metadata["filename"] == "file2.pdf"
            assert documents[1].metadata["score"] == 0.7

    def test_convert_file_results(self):
        """Test conversion of file search results."""
        with patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.get_from_dict_or_env") as mock_env:
            mock_env.return_value = "test-api-key"
            
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store1"],
                search_type="file"
            )
            
            chunk1 = MockChunkResult("Chunk 1 content")
            chunk2 = MockChunkResult("Chunk 2 content")
            
            mock_response = MockSearchResponse([
                MockFileResult("document.pdf", 0.8, "file_123", [chunk1, chunk2]),
            ])
            
            documents = retriever._convert_results_to_documents(mock_response)
            
            assert len(documents) == 1
            assert documents[0].page_content == "Chunk 1 content\n\nChunk 2 content"
            assert documents[0].metadata["filename"] == "document.pdf"
            assert documents[0].metadata["score"] == 0.8
            assert documents[0].metadata["file_id"] == "file_123"

    def test_convert_file_results_no_chunks(self):
        """Test conversion of file search results without chunks."""
        with patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.get_from_dict_or_env") as mock_env:
            mock_env.return_value = "test-api-key"
            
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store1"],
                search_type="file"
            )
            
            mock_response = MockSearchResponse([
                MockFileResult("document.pdf", 0.8, "file_123", []),
            ])
            
            documents = retriever._convert_results_to_documents(mock_response)
            
            assert len(documents) == 1
            assert documents[0].page_content == "[File: document.pdf - No chunks returned by API]"
            assert documents[0].metadata["filename"] == "document.pdf"

    @patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.Mixedbread")
    def test_get_relevant_documents_chunk_search(self, mock_mixedbread_class):
        """Test sync chunk search."""
        with patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.get_from_dict_or_env") as mock_env:
            mock_env.return_value = "test-api-key"
            
            # Setup mock client
            mock_client = Mock()
            mock_mixedbread_class.return_value = mock_client
            
            mock_response = MockSearchResponse([
                MockChunkResult("Test content", "test.pdf", 0.9),
            ])
            mock_client.vector_stores.search.return_value = mock_response
            
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store1"],
                search_type="chunk"
            )
            
            # Mock run manager
            run_manager = Mock()
            
            documents = retriever._get_relevant_documents("test query", run_manager=run_manager)
            
            assert len(documents) == 1
            assert documents[0].page_content == "Test content"
            
            # Verify API call
            mock_client.vector_stores.search.assert_called_once_with(
                query="test query",
                vector_store_identifiers=["store1"],
                top_k=10,
                search_options={"return_metadata": True},
            )

    @patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.AsyncMixedbread")
    @pytest.mark.asyncio
    async def test_aget_relevant_documents_file_search(self, mock_async_mixedbread_class):
        """Test async file search."""
        with patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.get_from_dict_or_env") as mock_env:
            mock_env.return_value = "test-api-key"
            
            # Setup mock async client
            mock_async_client = AsyncMock()
            mock_async_mixedbread_class.return_value = mock_async_client
            
            chunk = MockChunkResult("File chunk content")
            mock_response = MockSearchResponse([
                MockFileResult("test.pdf", 0.8, "file_123", [chunk]),
            ])
            mock_async_client.vector_stores.files.search.return_value = mock_response
            
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store1"],
                search_type="file",
                score_threshold=0.5
            )
            
            # Mock run manager
            run_manager = AsyncMock()
            
            documents = await retriever._aget_relevant_documents("test query", run_manager=run_manager)
            
            assert len(documents) == 1
            assert documents[0].page_content == "File chunk content"
            assert documents[0].metadata["file_id"] == "file_123"
            
            # Verify API call
            mock_async_client.vector_stores.files.search.assert_called_once_with(
                query="test query",
                vector_store_identifiers=["store1"],
                top_k=10,
                score_threshold=0.5,
                search_options={"return_metadata": True, "return_chunks": True},
            )

    @patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.Mixedbread")
    def test_empty_query_handling(self, mock_mixedbread_class):
        """Test handling of empty queries."""
        with patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.get_from_dict_or_env") as mock_env:
            mock_env.return_value = "test-api-key"
            
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store1"]
            )
            
            run_manager = Mock()
            
            # Empty string
            documents = retriever._get_relevant_documents("", run_manager=run_manager)
            assert documents == []
            
            # Whitespace only
            documents = retriever._get_relevant_documents("   ", run_manager=run_manager)
            assert documents == []

    @patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.Mixedbread")
    def test_error_handling(self, mock_mixedbread_class):
        """Test error handling during search."""
        with patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.get_from_dict_or_env") as mock_env:
            mock_env.return_value = "test-api-key"
            
            # Setup mock client that raises an exception
            mock_client = Mock()
            mock_mixedbread_class.return_value = mock_client
            mock_client.vector_stores.search.side_effect = Exception("API Error")
            
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store1"]
            )
            
            # Should return empty list on error
            # Mock run manager
            run_manager = Mock()
            
            documents = retriever._get_relevant_documents("test query", run_manager=run_manager)
            assert documents == []

    def test_secret_str_api_key(self):
        """Test SecretStr API key handling."""
        with patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.Mixedbread") as mock_mixedbread_class:
            retriever = MixedbreadVectorStoreRetriever(
                vector_store_identifiers=["store1"],
                api_key=SecretStr("secret-key")
            )
            
            # Verify client was initialized with resolved secret
            mock_mixedbread_class.assert_called_with(api_key="secret-key")

    def test_base_url_parameter(self):
        """Test base_url parameter."""
        with patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.get_from_dict_or_env") as mock_env:
            mock_env.return_value = "test-api-key"
            
            with patch("mixedbread_ai_langchain.retrievers.vector_store_retriever.Mixedbread") as mock_mixedbread_class:
                retriever = MixedbreadVectorStoreRetriever(
                    vector_store_identifiers=["store1"],
                    base_url="https://custom.api.com"
                )
                
                # Verify client was initialized with base_url
                mock_mixedbread_class.assert_called_with(
                    api_key="test-api-key",
                    base_url="https://custom.api.com"
                )