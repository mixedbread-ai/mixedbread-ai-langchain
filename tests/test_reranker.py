import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain_core.documents import Document
from mixedbread_ai_langchain.compressors.reranker import MixedbreadReranker


class TestMixedbreadReranker:
    """Lean test suite for MixedbreadReranker."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch.dict("os.environ", {"MXBAI_API_KEY": "test-key"}):
            reranker = MixedbreadReranker()
            assert reranker.model == "mixedbread-ai/mxbai-rerank-large-v2"
            assert reranker.top_k == 3
            assert reranker.return_input is True

    def test_init_with_parameters(self):
        """Test initialization with custom parameters."""
        reranker = MixedbreadReranker(
            model="custom-model",
            api_key="test-key",
            top_k=5,
            return_input=False,
        )
        assert reranker.model == "custom-model"
        assert reranker.top_k == 5
        assert reranker.return_input is False

    def test_init_fail_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="MXBAI_API_KEY"):
                MixedbreadReranker()

    @patch("mixedbread_ai_langchain.compressors.reranker.Mixedbread")
    def test_compress_documents_empty_input(self, mock_mixedbread):
        """Test compress_documents with empty input."""
        reranker = MixedbreadReranker(api_key="test-key")
        result = reranker.compress_documents([], "test query")

        assert result == []
        mock_mixedbread.return_value.rerank.assert_not_called()

    @patch("mixedbread_ai_langchain.compressors.reranker.Mixedbread")
    def test_compress_documents_empty_query(self, mock_mixedbread):
        """Test compress_documents with empty query."""
        documents = [
            Document(page_content="Document 1"),
            Document(page_content="Document 2"),
        ]
        
        reranker = MixedbreadReranker(api_key="test-key", top_k=1)
        result = reranker.compress_documents(documents, "")

        assert len(result) == 1
        assert result[0].page_content == "Document 1"
        mock_mixedbread.return_value.rerank.assert_not_called()

    @patch("mixedbread_ai_langchain.compressors.reranker.Mixedbread")
    def test_compress_documents_success(self, mock_mixedbread):
        """Test successful document compression."""
        # Mock response
        mock_result_1 = Mock(index=1, score=0.9)
        mock_result_2 = Mock(index=0, score=0.7)
        mock_response = Mock()
        mock_response.data = [mock_result_1, mock_result_2]
        mock_mixedbread.return_value.rerank.return_value = mock_response

        # Input documents
        documents = [
            Document(page_content="Document 1", metadata={"source": "doc1"}),
            Document(page_content="Document 2", metadata={"source": "doc2"}),
        ]

        reranker = MixedbreadReranker(api_key="test-key")
        result = reranker.compress_documents(documents, "test query")

        # Should return reranked documents
        assert len(result) == 2
        assert result[0].page_content == "Document 2"  # Higher score
        assert result[0].metadata["rerank_score"] == 0.9
        assert result[0].metadata["rerank_index"] == 1
        assert result[0].metadata["source"] == "doc2"  # Original metadata preserved

        assert result[1].page_content == "Document 1"
        assert result[1].metadata["rerank_score"] == 0.7
        assert result[1].metadata["rerank_index"] == 0

        mock_mixedbread.return_value.rerank.assert_called_once_with(
            model="mixedbread-ai/mxbai-rerank-large-v2",
            query="test query",
            input=["Document 1", "Document 2"],
            top_k=3,
            return_input=True,
        )

    @patch("mixedbread_ai_langchain.compressors.reranker.Mixedbread")
    def test_compress_documents_api_error(self, mock_mixedbread):
        """Test compress_documents handles API errors gracefully."""
        # Mock API error
        mock_mixedbread.return_value.rerank.side_effect = Exception("API Error")

        documents = [
            Document(page_content="Document 1"),
            Document(page_content="Document 2"),
        ]

        reranker = MixedbreadReranker(api_key="test-key", top_k=1)
        result = reranker.compress_documents(documents, "test query")

        # Should fallback to original order
        assert len(result) == 1
        assert result[0].page_content == "Document 1"

    @patch("mixedbread_ai_langchain.compressors.reranker.Mixedbread")
    def test_compress_documents_empty_response(self, mock_mixedbread):
        """Test compress_documents with empty API response."""
        # Mock empty response
        mock_response = Mock()
        mock_response.data = []
        mock_mixedbread.return_value.rerank.return_value = mock_response

        documents = [
            Document(page_content="Document 1"),
            Document(page_content="Document 2"),
        ]

        reranker = MixedbreadReranker(api_key="test-key")
        result = reranker.compress_documents(documents, "test query")

        # Should return original documents limited by top_k
        assert len(result) == 2
        assert result[0].page_content == "Document 1"
        assert result[1].page_content == "Document 2"

    @patch("mixedbread_ai_langchain.compressors.reranker.AsyncMixedbread")
    @pytest.mark.asyncio
    async def test_acompress_documents(self, mock_async_mixedbread):
        """Test async compress_documents method."""
        # Mock async response
        mock_result = Mock(index=0, score=0.9)
        mock_response = Mock()
        mock_response.data = [mock_result]
        mock_async_mixedbread.return_value.rerank = AsyncMock(return_value=mock_response)

        documents = [Document(page_content="Document 1")]

        reranker = MixedbreadReranker(api_key="test-key")
        result = await reranker.acompress_documents(documents, "test query")

        assert len(result) == 1
        assert result[0].page_content == "Document 1"
        assert result[0].metadata["rerank_score"] == 0.9
        mock_async_mixedbread.return_value.rerank.assert_called_once()

    @patch("mixedbread_ai_langchain.compressors.reranker.AsyncMixedbread")
    @pytest.mark.asyncio
    async def test_acompress_documents_error(self, mock_async_mixedbread):
        """Test async compress_documents handles errors gracefully."""
        # Mock async error
        mock_async_mixedbread.return_value.rerank = AsyncMock(side_effect=Exception("API Error"))

        documents = [Document(page_content="Document 1")]

        reranker = MixedbreadReranker(api_key="test-key")
        result = await reranker.acompress_documents(documents, "test query")

        # Should fallback to original order
        assert len(result) == 1
        assert result[0].page_content == "Document 1"