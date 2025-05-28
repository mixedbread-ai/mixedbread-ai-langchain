import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from mixedbread_ai_langchain import MixedbreadDocumentLoader
from .test_config import TestConfig
import tempfile
import os


DEFAULT_VALUES = {
    "chunking_strategy": "page",
    "return_format": "markdown",
    "element_types": ["text", "title", "list-item", "table"],
    "max_wait_time": 300,
    "poll_interval": 5,
}


class TestMixedbreadDocumentLoader:
    def test_init_default(self, monkeypatch):
        """
        Test default initialization parameters for MixedbreadDocumentLoader.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        loader = MixedbreadDocumentLoader("fake-file.pdf")

        assert len(loader.file_paths) == 1
        assert loader.file_paths[0].name == "fake-file.pdf"
        assert loader.chunking_strategy == DEFAULT_VALUES["chunking_strategy"]
        assert loader.return_format == DEFAULT_VALUES["return_format"]
        assert loader.element_types == DEFAULT_VALUES["element_types"]
        assert loader.max_wait_time == DEFAULT_VALUES["max_wait_time"]
        assert loader.poll_interval == DEFAULT_VALUES["poll_interval"]

    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for MixedbreadDocumentLoader.
        """
        loader = MixedbreadDocumentLoader(
            file_paths=["test.pdf", "test2.docx"],
            api_key="test-api-key",
            chunking_strategy="chunk",
            return_format="plain",
            element_types=["text"],
            max_wait_time=120,
            poll_interval=2,
        )

        assert len(loader.file_paths) == 2
        assert loader.file_paths[0].name == "test.pdf"
        assert loader.file_paths[1].name == "test2.docx"
        assert loader.chunking_strategy == "chunk"
        assert loader.return_format == "plain"
        assert loader.element_types == ["text"]
        assert loader.max_wait_time == 120
        assert loader.poll_interval == 2

    def test_init_fail_wo_api_key(self, monkeypatch):
        """
        Test that initialization fails when no API key is provided.
        """
        monkeypatch.delenv("MXBAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Mixedbread API key not found"):
            MixedbreadDocumentLoader("fake-file.pdf")

    @patch("mixedbread_ai_langchain.loaders.document_loaders.MixedbreadClient")
    def test_upload_file_success(self, mock_client_class):
        """
        Test successful file upload.
        """
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            temp_file = f.name

        try:
            # Mock the client and response
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_result = Mock()
            mock_result.id = "test-file-id"
            mock_client.client.files.create.return_value = mock_result

            loader = MixedbreadDocumentLoader(
                file_paths=[temp_file], api_key="fake-api-key"
            )

            file_id = loader._upload_file(loader.file_paths[0])
            assert file_id == "test-file-id"
            mock_client.client.files.create.assert_called_once()
        finally:
            # Clean up the temporary file
            os.unlink(temp_file)

    @patch("mixedbread_ai_langchain.loaders.document_loaders.MixedbreadClient")
    def test_create_parsing_job_success(self, mock_client_class):
        """
        Test successful parsing job creation.
        """
        # Mock the client and response
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_result = Mock()
        mock_result.id = "test-job-id"
        mock_client.client.parsing.jobs.create.return_value = mock_result

        loader = MixedbreadDocumentLoader(
            file_paths=["test.pdf"], api_key="fake-api-key"
        )

        job_id = loader._create_parsing_job("test-file-id")
        assert job_id == "test-job-id"
        mock_client.client.parsing.jobs.create.assert_called_once()

    def test_load_with_empty_file_list(self):
        """
        Test load() with empty file list.
        """
        # This should work but return empty list since no files to process
        loader = MixedbreadDocumentLoader(file_paths=[], api_key="fake-api-key")
        documents = loader.load()
        assert documents == []

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_upload_and_parse_file(self):
        """
        Test uploading and parsing a real file.
        Note: This test requires a valid API key and will make real API calls.
        """
        # Create a temporary text file for testing
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document for parsing.\n\nIt has multiple lines.")
            temp_file = f.name

        try:
            loader_config = TestConfig.get_test_document_loader_config()
            loader = MixedbreadDocumentLoader(file_paths=[temp_file], **loader_config)

            # This test might take a while since it uploads, parses, and waits for completion
            documents = loader.load()

            # Basic assertions that work regardless of content
            assert isinstance(documents, list)
            for doc in documents:
                assert isinstance(doc, Document)
                assert hasattr(doc, "page_content")
                assert hasattr(doc, "metadata")
                assert isinstance(doc.metadata, dict)
        finally:
            # Clean up the temporary file
            os.unlink(temp_file)
