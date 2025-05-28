import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from ..common.client import MixedbreadClient


class MixedbreadDocumentLoader(BaseLoader):
    """
    This loader uploads files to Mixedbread AI, creates parsing jobs, waits for completion,
    and converts the parsed results into LangChain Document objects. It supports various
    file formats and provides configurable parsing options.
    """

    def __init__(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        api_key: Optional[str] = None,
        chunking_strategy: Optional[str] = "page",
        return_format: Literal["markdown", "plain"] = "markdown",
        element_types: Optional[List[str]] = None,
        max_wait_time: int = 300,
        poll_interval: int = 5,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
    ):
        """
        Initialize the Mixedbread document loader.

        Args:
            file_paths: Path(s) to the file(s) to parse. Can be a single path or list of paths.
            api_key: API key for Mixedbread AI (or set MXBAI_API_KEY env var).
            chunking_strategy: Strategy for chunking the document content.
            return_format: Format for the returned content ("markdown" or "plain").
            element_types: List of element types to extract (e.g., ["text", "title", "list-item", "table"]).
            max_wait_time: Maximum time to wait for parsing job completion (seconds).
            poll_interval: Interval between polling for job status (seconds).
            base_url: Base URL for the API.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
        """
        if isinstance(file_paths, (str, Path)):
            self.file_paths = [Path(file_paths)]
        else:
            self.file_paths = [Path(fp) for fp in file_paths]

        self.chunking_strategy = chunking_strategy
        self.return_format = return_format
        self.element_types = element_types or ["text", "title", "list-item", "table"]
        self.max_wait_time = max_wait_time
        self.poll_interval = poll_interval

        self._client = MixedbreadClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _upload_file(self, file_path: Path) -> str:
        """
        Upload a file to Mixedbread AI.

        Args:
            file_path: Path to the file to upload.

        Returns:
            The file ID assigned by the API.

        Raises:
            Exception: If file upload fails.
        """
        with open(file_path, "rb") as f:
            result = self._client.client.files.create(file=f)
        return result.id

    def _create_parsing_job(self, file_id: str) -> str:
        """
        Create a parsing job for the uploaded file.

        Args:
            file_id: ID of the uploaded file.

        Returns:
            The parsing job ID.

        Raises:
            Exception: If job creation fails.
        """
        result = self._client.client.parsing.jobs.create(
            file_id=file_id,
            chunking_strategy=self.chunking_strategy,
            return_format=self.return_format,
            element_types=self.element_types,
        )
        return result.id

    def _wait_for_job_completion(self, job_id: str) -> Dict[str, Any]:
        """
        Wait for the parsing job to complete and return the result.

        Args:
            job_id: ID of the parsing job.

        Returns:
            The parsing job result as a dictionary.

        Raises:
            RuntimeError: If the parsing job fails.
            TimeoutError: If the job doesn't complete within max_wait_time.
        """
        start_time = time.time()

        while time.time() - start_time < self.max_wait_time:
            result = self._client.client.parsing.jobs.retrieve(job_id=job_id)

            if result.status == "completed":
                return result.model_dump()
            elif result.status == "failed":
                error_msg = result.error or "Unknown parsing error"
                raise RuntimeError(f"Parsing job failed: {error_msg}")

            time.sleep(self.poll_interval)

        raise TimeoutError(
            f"Parsing job {job_id} did not complete within {self.max_wait_time} seconds"
        )

    def _create_documents_from_result(
        self,
        parsing_result: Dict[str, Any],
        source_path: Path,
        meta: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Convert parsing results into LangChain Document objects.

        Args:
            parsing_result: The parsing job result from the API.
            source_path: Path to the source file.
            meta: Additional metadata to include in documents.

        Returns:
            List of LangChain Document objects.
        """
        documents = []
        result_data = parsing_result.get("result", {})
        chunks = result_data.get("chunks", [])

        base_metadata = {
            "file_path": source_path.name,
            "parsing_job_id": parsing_result.get("id"),
            "chunking_strategy": result_data.get("chunking_strategy"),
            "return_format": result_data.get("return_format"),
            "element_types": result_data.get("element_types"),
            "page_sizes": result_data.get("page_sizes"),
            "total_chunks": len(chunks),
        }

        if meta:
            base_metadata.update(meta)

        for i, chunk in enumerate(chunks):
            chunk_content = chunk.get("content", "")
            content_to_embed = chunk.get("content_to_embed", chunk_content)
            elements = chunk.get("elements", [])

            chunk_metadata = base_metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": i,
                    "elements": elements,
                    "element_count": len(elements),
                    "content_to_embed": content_to_embed,
                }
            )

            if elements:
                element_types_in_chunk = list(
                    set(elem.get("type") for elem in elements)
                )
                chunk_metadata["element_types_in_chunk"] = element_types_in_chunk

                pages = list(
                    set(
                        elem.get("page")
                        for elem in elements
                        if elem.get("page") is not None
                    )
                )
                if pages:
                    chunk_metadata["pages"] = sorted(pages)
                    chunk_metadata["page_range"] = (
                        f"{min(pages)}-{max(pages)}"
                        if len(pages) > 1
                        else str(pages[0])
                    )

            document = Document(page_content=chunk_content, metadata=chunk_metadata)
            documents.append(document)

        return documents

    def _process_single_file(
        self, file_path: Path, meta: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process a single file through the complete parsing pipeline.

        Args:
            file_path: Path to the file to process.
            meta: Additional metadata to include in documents.

        Returns:
            List of Document objects created from the file.
        """
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            file_id = self._upload_file(file_path)
            job_id = self._create_parsing_job(file_id)
            parsing_result = self._wait_for_job_completion(job_id)
            documents = self._create_documents_from_result(
                parsing_result, file_path, meta
            )
            return documents

        except Exception as e:
            error_msg = f"Failed to parse {file_path}: {str(e)}"

            error_metadata = {
                "file_path": file_path.name,
                "parsing_error": error_msg,
                "parsing_status": "failed",
            }

            if meta:
                error_metadata.update(meta)

            return [Document(page_content="", metadata=error_metadata)]

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazy loader that yields documents one by one.

        This method processes files sequentially and yields documents as they are created,
        which is memory-efficient for large numbers of files.

        Yields:
            Document objects one at a time.
        """
        for file_path in self.file_paths:
            documents = self._process_single_file(file_path)
            for doc in documents:
                yield doc

    def load(self) -> List[Document]:
        """
        Load all documents at once.

        This method processes all files and returns a complete list of documents.

        Returns:
            List of all Document objects from all processed files.
        """
        return list(self.lazy_load())
