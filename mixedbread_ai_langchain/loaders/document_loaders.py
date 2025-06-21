import time
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Union
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from pydantic import SecretStr
from mixedbread import Mixedbread


class MixedbreadDocumentLoader(BaseLoader):
    """
    Mixedbread document loader integration for LangChain.

    Implementation that uploads files to Mixedbread API, creates parsing jobs,
    and converts the parsed results into LangChain Document objects.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        api_key: Union[SecretStr, str, None] = None,
        base_url: Optional[str] = None,
        chunking_strategy: Optional[str] = "page",
        return_format: Literal["markdown", "plain"] = "markdown",
        element_types: Optional[List[str]] = None,
        max_wait_time: int = 300,
        poll_interval: int = 5,
    ):
        """
        Initialize the Mixedbread document loader.

        Args:
            file_path: Path to the file to parse.
            api_key: API key for Mixedbread AI (or set MXBAI_API_KEY env var).
            base_url: Base URL for the API.
            chunking_strategy: Strategy for chunking the document content.
            return_format: Format for the returned content ("markdown" or "plain").
            element_types: List of element types to extract.
            max_wait_time: Maximum time to wait for parsing job completion (seconds).
            poll_interval: Interval between polling for job status (seconds).
        """
        if api_key is None:
            api_key = get_from_dict_or_env({}, "api_key", "MXBAI_API_KEY")

        if isinstance(api_key, str):
            api_key = SecretStr(api_key)

        self.file_path = Path(file_path)
        self.api_key = api_key
        self.chunking_strategy = chunking_strategy
        self.return_format = return_format
        self.element_types = element_types or ["text", "title", "list-item", "table"]
        self.max_wait_time = max_wait_time
        self.poll_interval = poll_interval

        # Initialize client
        resolved_api_key = self.api_key.get_secret_value()
        client_kwargs = {"api_key": resolved_api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = Mixedbread(**client_kwargs)

    def _upload_file(self) -> str:
        """Upload the file to Mixedbread AI and return file ID."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with open(self.file_path, "rb") as f:
            result = self._client.files.create(file=f)
        return result.id

    def _create_parsing_job(self, file_id: str) -> str:
        """Create a parsing job and return job ID."""
        result = self._client.parsing.jobs.create(
            file_id=file_id,
            chunking_strategy=self.chunking_strategy,
            return_format=self.return_format,
            element_types=self.element_types,
        )
        return result.id

    def _wait_for_completion(self, job_id: str) -> dict:
        """Wait for parsing job completion and return results."""
        start_time = time.time()

        while time.time() - start_time < self.max_wait_time:
            result = self._client.parsing.jobs.retrieve(job_id=job_id)

            if result.status == "completed":
                return result.model_dump()
            elif result.status == "failed":
                error_msg = getattr(result, "error", "Unknown parsing error")
                raise RuntimeError(f"Parsing job failed: {error_msg}")

            time.sleep(self.poll_interval)

        raise TimeoutError(
            f"Parsing job {job_id} did not complete within {self.max_wait_time} seconds"
        )

    def _create_documents(self, parsing_result: dict) -> List[Document]:
        """Convert parsing results to LangChain Documents."""
        documents = []
        result_data = parsing_result.get("result", {})
        chunks = result_data.get("chunks", [])

        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")

            # Create minimal metadata
            metadata = {
                "source": str(self.file_path),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "parsing_job_id": parsing_result.get("id"),
            }

            # Add element info if available
            elements = chunk.get("elements", [])
            if elements:
                # Extract page numbers if available
                pages = {
                    elem.get("page")
                    for elem in elements
                    if elem.get("page") is not None
                }
                if pages:
                    metadata["pages"] = sorted(pages)

            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def load(self) -> List[Document]:
        """
        Load and parse the document.

        Returns:
            List of Document objects created from the parsed file.
        """
        try:
            # Execute the parsing workflow
            file_id = self._upload_file()
            job_id = self._create_parsing_job(file_id)
            parsing_result = self._wait_for_completion(job_id)
            documents = self._create_documents(parsing_result)

            return documents

        except Exception as e:
            # Return error document on failure
            error_content = f"Failed to parse {self.file_path}: {str(e)}"
            error_metadata = {
                "source": str(self.file_path),
                "parsing_error": True,
                "error_message": str(e),
            }
            return [Document(page_content=error_content, metadata=error_metadata)]

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazy load documents one by one.

        Yields:
            Document objects one at a time.
        """
        documents = self.load()
        for doc in documents:
            yield doc
