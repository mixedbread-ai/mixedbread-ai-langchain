import pytest
from unittest.mock import Mock, AsyncMock, patch
from mixedbread_ai_langchain.embedders.embeddings import MixedbreadEmbeddings


class TestMixedbreadEmbeddings:
    """Lean test suite for MixedbreadEmbeddings."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch.dict("os.environ", {"MXBAI_API_KEY": "test-key"}):
            embeddings = MixedbreadEmbeddings()
            assert embeddings.model == "mixedbread-ai/mxbai-embed-large-v1"
            assert embeddings.normalized is True
            assert embeddings.encoding_format == "float"

    def test_init_with_parameters(self):
        """Test initialization with custom parameters."""
        embeddings = MixedbreadEmbeddings(
            model="custom-model",
            api_key="test-key",
            normalized=False,
            encoding_format="float16",
            dimensions=512,
        )
        assert embeddings.model == "custom-model"
        assert embeddings.normalized is False
        assert embeddings.encoding_format == "float16"
        assert embeddings.dimensions == 512

    def test_init_fail_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="MXBAI_API_KEY"):
                MixedbreadEmbeddings()

    @patch("mixedbread_ai_langchain.embedders.embeddings.Mixedbread")
    def test_embed_query(self, mock_mixedbread):
        """Test embed_query method."""
        # Mock response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_mixedbread.return_value.embed.return_value = mock_response

        embeddings = MixedbreadEmbeddings(api_key="test-key")
        result = embeddings.embed_query("test query")

        assert result == [0.1, 0.2, 0.3]
        mock_mixedbread.return_value.embed.assert_called_once()

    @patch("mixedbread_ai_langchain.embedders.embeddings.Mixedbread")
    def test_embed_query_empty_text(self, mock_mixedbread):
        """Test embed_query with empty text."""
        embeddings = MixedbreadEmbeddings(api_key="test-key")
        result = embeddings.embed_query("")

        assert result == []
        mock_mixedbread.return_value.embed.assert_not_called()

    @patch("mixedbread_ai_langchain.embedders.embeddings.Mixedbread")
    def test_embed_documents(self, mock_mixedbread):
        """Test embed_documents method."""
        # Mock response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_mixedbread.return_value.embed.return_value = mock_response

        embeddings = MixedbreadEmbeddings(api_key="test-key")
        result = embeddings.embed_documents(["doc1", "doc2"])

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_mixedbread.return_value.embed.assert_called_once()

    @patch("mixedbread_ai_langchain.embedders.embeddings.Mixedbread")
    def test_embed_documents_with_empty_texts(self, mock_mixedbread):
        """Test embed_documents handles empty texts correctly."""
        # Mock response for only non-empty text
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_mixedbread.return_value.embed.return_value = mock_response

        embeddings = MixedbreadEmbeddings(api_key="test-key")
        result = embeddings.embed_documents(["doc1", "", "doc3"])

        # Should return embeddings with empty list for empty text
        assert len(result) == 3
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == []
        assert result[2] == []

    @patch("mixedbread_ai_langchain.embedders.embeddings.AsyncMixedbread")
    @pytest.mark.asyncio
    async def test_aembed_query(self, mock_async_mixedbread):
        """Test async embed_query method."""
        # Mock async response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_async_mixedbread.return_value.embed = AsyncMock(return_value=mock_response)

        embeddings = MixedbreadEmbeddings(api_key="test-key")
        result = await embeddings.aembed_query("test query")

        assert result == [0.1, 0.2, 0.3]
        mock_async_mixedbread.return_value.embed.assert_called_once()

    @patch("mixedbread_ai_langchain.embedders.embeddings.AsyncMixedbread")
    @pytest.mark.asyncio
    async def test_aembed_documents(self, mock_async_mixedbread):
        """Test async embed_documents method."""
        # Mock async response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_async_mixedbread.return_value.embed = AsyncMock(return_value=mock_response)

        embeddings = MixedbreadEmbeddings(api_key="test-key")
        result = await embeddings.aembed_documents(["doc1", "doc2"])

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_async_mixedbread.return_value.embed.assert_called_once()