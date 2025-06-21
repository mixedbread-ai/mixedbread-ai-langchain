import pytest
from unittest.mock import Mock, patch, mock_open
from mixedbread_ai_langchain.loaders.document_loaders import MixedbreadDocumentLoader


class TestMixedbreadDocumentLoader:
    """Lean test suite for MixedbreadDocumentLoader."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch.dict("os.environ", {"MXBAI_API_KEY": "test-key"}):
            loader = MixedbreadDocumentLoader("test.pdf")
            assert str(loader.file_path) == "test.pdf"
            assert loader.chunking_strategy == "page"
            assert loader.return_format == "markdown"
            assert loader.max_wait_time == 300
            assert loader.poll_interval == 5

    def test_init_with_parameters(self):
        """Test initialization with custom parameters."""
        loader = MixedbreadDocumentLoader(
            file_path="document.pdf",
            api_key="test-key",
            chunking_strategy="paragraph",
            return_format="plain",
            element_types=["text"],
            max_wait_time=600,
            poll_interval=10,
        )
        assert str(loader.file_path) == "document.pdf"
        assert loader.chunking_strategy == "paragraph"
        assert loader.return_format == "plain"
        assert loader.element_types == ["text"]
        assert loader.max_wait_time == 600
        assert loader.poll_interval == 10

    def test_init_fail_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="MXBAI_API_KEY"):
                MixedbreadDocumentLoader("test.pdf")

    @patch("mixedbread_ai_langchain.loaders.document_loaders.Mixedbread")
    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake pdf content")
    def test_upload_file_success(self, mock_file, mock_exists, mock_mixedbread):
        """Test successful file upload."""
        mock_exists.return_value = True
        mock_response = Mock()
        mock_response.id = "file123"
        mock_mixedbread.return_value.files.create.return_value = mock_response

        loader = MixedbreadDocumentLoader("test.pdf", api_key="test-key")
        file_id = loader._upload_file()

        assert file_id == "file123"
        mock_mixedbread.return_value.files.create.assert_called_once()

    @patch("pathlib.Path.exists")
    def test_upload_file_not_found(self, mock_exists):
        """Test file upload with missing file."""
        mock_exists.return_value = False

        loader = MixedbreadDocumentLoader("missing.pdf", api_key="test-key")

        with pytest.raises(FileNotFoundError, match="File not found"):
            loader._upload_file()

    @patch("mixedbread_ai_langchain.loaders.document_loaders.Mixedbread")
    def test_create_parsing_job_success(self, mock_mixedbread):
        """Test successful parsing job creation."""
        mock_response = Mock()
        mock_response.id = "job123"
        mock_mixedbread.return_value.parsing.jobs.create.return_value = mock_response

        loader = MixedbreadDocumentLoader("test.pdf", api_key="test-key")
        job_id = loader._create_parsing_job("file123")

        assert job_id == "job123"
        mock_mixedbread.return_value.parsing.jobs.create.assert_called_once_with(
            file_id="file123",
            chunking_strategy="page",
            return_format="markdown",
            element_types=["text", "title", "list-item", "table"],
        )

    @patch("mixedbread_ai_langchain.loaders.document_loaders.Mixedbread")
    @patch("time.sleep")
    def test_wait_for_completion_success(self, mock_sleep, mock_mixedbread):
        """Test successful job completion waiting."""
        # Mock completed job result
        mock_result = Mock()
        mock_result.status = "completed"
        mock_result.model_dump.return_value = {"id": "job123", "status": "completed"}
        mock_mixedbread.return_value.parsing.jobs.retrieve.return_value = mock_result

        loader = MixedbreadDocumentLoader("test.pdf", api_key="test-key")
        result = loader._wait_for_completion("job123")

        assert result == {"id": "job123", "status": "completed"}
        mock_mixedbread.return_value.parsing.jobs.retrieve.assert_called_with(
            job_id="job123"
        )

    @patch("mixedbread_ai_langchain.loaders.document_loaders.Mixedbread")
    @patch("time.sleep")
    def test_wait_for_completion_failed(self, mock_sleep, mock_mixedbread):
        """Test job completion with failed status."""
        mock_result = Mock()
        mock_result.status = "failed"
        mock_result.error = "Parsing error occurred"
        mock_mixedbread.return_value.parsing.jobs.retrieve.return_value = mock_result

        loader = MixedbreadDocumentLoader("test.pdf", api_key="test-key")

        with pytest.raises(
            RuntimeError, match="Parsing job failed: Parsing error occurred"
        ):
            loader._wait_for_completion("job123")

    @patch("mixedbread_ai_langchain.loaders.document_loaders.Mixedbread")
    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_completion_timeout(self, mock_time, mock_sleep, mock_mixedbread):
        """Test job completion timeout."""
        # Mock time progression to trigger timeout
        mock_time.side_effect = [0, 301]  # Start time, then past max_wait_time

        mock_result = Mock()
        mock_result.status = "pending"
        mock_mixedbread.return_value.parsing.jobs.retrieve.return_value = mock_result

        loader = MixedbreadDocumentLoader("test.pdf", api_key="test-key")

        with pytest.raises(TimeoutError, match="did not complete within 300 seconds"):
            loader._wait_for_completion("job123")

    def test_create_documents_success(self):
        """Test successful document creation from parsing results."""
        parsing_result = {
            "id": "job123",
            "result": {
                "chunks": [
                    {
                        "content": "This is chunk 1",
                        "elements": [{"page": 1}, {"page": 2}],
                    },
                    {"content": "This is chunk 2", "elements": [{"page": 2}]},
                ]
            },
        }

        loader = MixedbreadDocumentLoader("test.pdf", api_key="test-key")
        documents = loader._create_documents(parsing_result)

        assert len(documents) == 2

        # Check first document
        assert documents[0].page_content == "This is chunk 1"
        assert documents[0].metadata["source"] == "test.pdf"
        assert documents[0].metadata["chunk_index"] == 0
        assert documents[0].metadata["total_chunks"] == 2
        assert documents[0].metadata["parsing_job_id"] == "job123"
        assert documents[0].metadata["pages"] == [1, 2]

        # Check second document
        assert documents[1].page_content == "This is chunk 2"
        assert documents[1].metadata["chunk_index"] == 1
        assert documents[1].metadata["pages"] == [2]

    def test_create_documents_empty_result(self):
        """Test document creation with empty parsing result."""
        parsing_result = {"id": "job123", "result": {"chunks": []}}

        loader = MixedbreadDocumentLoader("test.pdf", api_key="test-key")
        documents = loader._create_documents(parsing_result)

        assert documents == []

    @patch.object(MixedbreadDocumentLoader, "_upload_file")
    @patch.object(MixedbreadDocumentLoader, "_create_parsing_job")
    @patch.object(MixedbreadDocumentLoader, "_wait_for_completion")
    @patch.object(MixedbreadDocumentLoader, "_create_documents")
    def test_load_success(
        self, mock_create_docs, mock_wait, mock_create_job, mock_upload
    ):
        """Test successful complete loading workflow."""
        # Mock the workflow steps
        mock_upload.return_value = "file123"
        mock_create_job.return_value = "job123"
        mock_wait.return_value = {"id": "job123", "status": "completed"}
        mock_create_docs.return_value = [Mock(page_content="Test content")]

        loader = MixedbreadDocumentLoader("test.pdf", api_key="test-key")
        documents = loader.load()

        assert len(documents) == 1
        mock_upload.assert_called_once()
        mock_create_job.assert_called_once_with("file123")
        mock_wait.assert_called_once_with("job123")

    @patch.object(MixedbreadDocumentLoader, "_upload_file")
    def test_load_error_handling(self, mock_upload):
        """Test load method error handling."""
        mock_upload.side_effect = Exception("Upload failed")

        loader = MixedbreadDocumentLoader("test.pdf", api_key="test-key")
        documents = loader.load()

        assert len(documents) == 1
        assert "Failed to parse test.pdf: Upload failed" in documents[0].page_content
        assert documents[0].metadata["parsing_error"] is True
        assert documents[0].metadata["error_message"] == "Upload failed"

    @patch.object(MixedbreadDocumentLoader, "load")
    def test_lazy_load(self, mock_load):
        """Test lazy load method."""
        mock_docs = [Mock(page_content="Doc 1"), Mock(page_content="Doc 2")]
        mock_load.return_value = mock_docs

        loader = MixedbreadDocumentLoader("test.pdf", api_key="test-key")
        lazy_docs = list(loader.lazy_load())

        assert len(lazy_docs) == 2
        assert lazy_docs == mock_docs
        mock_load.assert_called_once()
