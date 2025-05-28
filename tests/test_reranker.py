import pytest
from langchain_core.documents import Document
from mixedbread_ai_langchain import MixedbreadReranker
from .test_config import TestConfig


DEFAULT_VALUES = {
    "model": "mixedbread-ai/mxbai-rerank-large-v1",  # This is the actual default in the code
    "top_k": 3,
    "return_input": True,
}


class TestMixedbreadReranker:
    def test_init_default(self, monkeypatch):
        """
        Test default initialization parameters for MixedbreadReranker.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "test-api-key")
        component = MixedbreadReranker()

        assert component.model == DEFAULT_VALUES["model"]
        assert component.top_k == DEFAULT_VALUES["top_k"]
        assert component.return_input == DEFAULT_VALUES["return_input"]

    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for MixedbreadReranker.
        """
        component = MixedbreadReranker(
            api_key="test-api-key",
            model="custom-model",
            top_k=5,
            return_input=False,
        )

        assert component.model == "custom-model"
        assert component.top_k == 5
        assert component.return_input is False

    def test_init_fail_wo_api_key(self, monkeypatch):
        """
        Test that initialization fails when no API key is provided.
        """
        monkeypatch.delenv("MXBAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Mixedbread API key not found"):
            MixedbreadReranker()

    def test_compress_documents_empty_input(self):
        """
        Test reranking with empty document list.
        """
        ranker = MixedbreadReranker(api_key="fake-api-key")
        result = ranker.compress_documents(documents=[], query="test query")
        assert result == []

    def test_compress_documents_empty_query(self):
        """
        Test reranking with empty query should return top_k documents.
        """
        ranker = MixedbreadReranker(api_key="fake-api-key", top_k=2)
        documents = [
            Document(page_content="Document 1"),
            Document(page_content="Document 2"),
            Document(page_content="Document 3"),
        ]
        result = ranker.compress_documents(documents=documents, query="")
        assert len(result) == 2  # Should return top_k documents

    def test_prepare_documents_for_reranking(self):
        """
        Test document preparation for reranking.
        """
        ranker = MixedbreadReranker(api_key="fake-api-key")
        documents = [
            Document(page_content="Content 1"),
            Document(page_content="Content 2"),
            Document(page_content=""),  # Empty content
        ]

        prepared = ranker._prepare_documents_for_reranking(documents)
        assert prepared == ["Content 1", "Content 2", ""]

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_basic_reranking(self):
        """
        Test basic reranking with real API call.
        """
        reranker_config = TestConfig.get_test_reranker_config()
        reranker = MixedbreadReranker(top_k=2, **reranker_config)

        documents = [
            Document(page_content="Paris is the capital of France"),
            Document(page_content="Berlin is the capital of Germany"),
            Document(page_content="Madrid is the capital of Spain"),
        ]

        result = reranker.compress_documents(
            documents=documents, query="What is the capital of Germany?"
        )

        assert isinstance(result, list)
        assert len(result) <= 2  # top_k = 2
        assert all(isinstance(doc, Document) for doc in result)
        assert all("rerank_score" in doc.metadata for doc in result)
        assert all("rerank_index" in doc.metadata for doc in result)
        assert all("rerank_model" in doc.metadata for doc in result)
        assert all("original_index" in doc.metadata for doc in result)

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    async def test_integration_async_reranking(self):
        """
        Test async reranking with real API call.
        """
        reranker_config = TestConfig.get_test_reranker_config()
        reranker = MixedbreadReranker(top_k=2, **reranker_config)

        documents = [
            Document(page_content="Paris is the capital of France"),
            Document(page_content="Berlin is the capital of Germany"),
            Document(page_content="Madrid is the capital of Spain"),
        ]

        result = await reranker.acompress_documents(
            documents=documents, query="What is the capital of Germany?"
        )

        assert isinstance(result, list)
        assert len(result) <= 2  # top_k = 2
        assert all(isinstance(doc, Document) for doc in result)
        assert all("rerank_score" in doc.metadata for doc in result)

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_reranking_with_metadata(self):
        """
        Test reranking preserves existing document metadata.
        """
        reranker_config = TestConfig.get_test_reranker_config()
        reranker = MixedbreadReranker(top_k=3, **reranker_config)

        documents = [
            Document(
                page_content="Paris is the capital of France",
                metadata={"country": "France", "type": "capital"},
            ),
            Document(
                page_content="Berlin is the capital of Germany",
                metadata={"country": "Germany", "type": "capital"},
            ),
            Document(
                page_content="Madrid is the capital of Spain",
                metadata={"country": "Spain", "type": "capital"},
            ),
        ]

        result = reranker.compress_documents(
            documents=documents, query="German capital city"
        )

        assert len(result) > 0
        for doc in result:
            # Original metadata should be preserved
            assert "country" in doc.metadata
            assert "type" in doc.metadata
            # Reranking metadata should be added
            assert "rerank_score" in doc.metadata
            assert "rerank_index" in doc.metadata
            assert "rerank_model" in doc.metadata
            assert "original_index" in doc.metadata
