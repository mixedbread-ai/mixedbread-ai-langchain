import pytest
from langchain_core.documents import Document
from mixedbread_ai_langchain import MixedbreadEmbeddings
from .test_config import TestConfig


DEFAULT_VALUES = {
    "model": "mixedbread-ai/mxbai-embed-large-v1",
    "normalized": True,
    "encoding_format": "float",
    "dimensions": None,
    "prompt": None,
}


class TestMixedbreadEmbeddings:
    def test_init_default(self, monkeypatch):
        """
        Test default initialization parameters for MixedbreadEmbeddings.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        embedder = MixedbreadEmbeddings()

        assert embedder.model == DEFAULT_VALUES["model"]
        assert embedder.normalized == DEFAULT_VALUES["normalized"]
        assert embedder.encoding_format.value == DEFAULT_VALUES["encoding_format"]
        assert embedder.dimensions == DEFAULT_VALUES["dimensions"]
        assert embedder.prompt == DEFAULT_VALUES["prompt"]

    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for MixedbreadEmbeddings.
        """
        embedder = MixedbreadEmbeddings(
            api_key="test-api-key",
            model="custom-model",
            normalized=False,
            encoding_format="binary",
            dimensions=500,
            prompt="test prompt",
        )

        assert embedder.model == "custom-model"
        assert not embedder.normalized
        assert embedder.encoding_format.value == "binary"
        assert embedder.dimensions == 500
        assert embedder.prompt == "test prompt"

    def test_init_fail_wo_api_key(self, monkeypatch):
        """
        Test that initialization fails when no API key is provided.
        """
        monkeypatch.delenv("MXBAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Mixedbread API key not found"):
            MixedbreadEmbeddings()

    def test_embed_documents_empty_input(self):
        """
        Test embedding with empty document list.
        """
        embedder = MixedbreadEmbeddings(api_key="fake-api-key")
        result = embedder.embed_documents([])
        assert result == []

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_embed_query(self):
        """
        Test basic query embedding with real API call.
        """
        embedder_config = TestConfig.get_test_embedder_config()
        embedder = MixedbreadEmbeddings(**embedder_config)

        query = "What is the capital of France?"
        embedding = embedder.embed_query(query)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_embed_documents(self):
        """
        Test basic document embedding with real API call.
        """
        embedder_config = TestConfig.get_test_embedder_config()
        embedder = MixedbreadEmbeddings(**embedder_config)

        texts = [
            "The Eiffel Tower is in Paris",
            "Machine learning is transforming industries",
        ]

        embeddings = embedder.embed_documents(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    async def test_integration_aembed_query(self):
        """
        Test async query embedding with real API call.
        """
        embedder_config = TestConfig.get_test_embedder_config()
        embedder = MixedbreadEmbeddings(**embedder_config)

        query = "What is the capital of France?"
        embedding = await embedder.aembed_query(query)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    async def test_integration_aembed_documents(self):
        """
        Test async document embedding with real API call.
        """
        embedder_config = TestConfig.get_test_embedder_config()
        embedder = MixedbreadEmbeddings(**embedder_config)

        texts = [
            "The Eiffel Tower is in Paris",
            "Machine learning is transforming industries",
        ]

        embeddings = await embedder.aembed_documents(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, float) for x in embedding)
