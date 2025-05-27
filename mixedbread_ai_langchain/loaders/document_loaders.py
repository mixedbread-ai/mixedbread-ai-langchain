import time
import asyncio
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from ..common.client import MixedbreadClient


class MixedbreadDocumentLoader(BaseLoader):
    """
    Document loader that uses Mixedbread AI's parsing service to parse various file types.
    """

    def __init__(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        api_key: Optional[str] = None,
        chunking_strategy: Literal["page", "paragraph", "sentence"] = "page",
        return_format: Literal["markdown", "text"] = "markdown",
        element_types: Optional[List[str]] = None,
        include_page_breaks: bool = True,
        max_wait_time: int = 300,
        poll_interval: int = 5,
        store_full_path: bool = False,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
        **kwargs: Any,
    ):
        """
        Initialize the Mixedbread document loader.
        """
        if isinstance(file_paths, (str, Path)):
            self.file_paths = [Path(file_paths)]
        else:
            self.file_paths = [Path(fp) for fp in file_paths]

        self.chunking_strategy = chunking_strategy
        self.return_format = return_format
        self.element_types = element_types or ["text", "title", "list-item", "table"]
        self.include_page_breaks = include_page_breaks
        self.max_wait_time = max_wait_time
        self.poll_interval = poll_interval
        self.store_full_path = store_full_path

        self._client = MixedbreadClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _upload_file(self, file_path: Path) -> str:
        """Upload a file to Mixedbread AI and return the file ID."""
        try:
            with open(file_path, "rb") as f:
                result = self._client.client.files.create(file=f)
            return result.id
        except Exception as e:
            raise RuntimeError(f"Failed to upload file {file_path}: {str(e)}") from e

    def _create_parsing_job(self, file_id: str) -> str:
        """Create a parsing job and return the job ID."""
        try:
            result = self._client.client.parsing.jobs.create(
                file_id=file_id,
                chunking_strategy=self.chunking_strategy,
                return_format=self.return_format,
                element_types=self.element_types,
            )
            return result.id
        except Exception as e:
            raise RuntimeError(f"Failed to create parsing job: {str(e)}") from e

    def _wait_for_job_completion(self, job_id: str) -> Dict[str, Any]:
        """Wait for the parsing job to complete and return the result."""
        start_time = time.time()

        while time.time() - start_time < self.max_wait_time:
            try:
                result = self._client.client.parsing.jobs.retrieve(job_id=job_id)

                if result.status == "completed":
                    return result.model_dump()
                elif result.status == "failed":
                    error_msg = getattr(result, "error", "Unknown parsing error")
                    raise RuntimeError(f"Parsing job failed: {error_msg}")

                time.sleep(self.poll_interval)

            except Exception as e:
                if "failed" in str(e).lower():
                    raise
                time.sleep(self.poll_interval)

        raise TimeoutError(
            f"Parsing job {job_id} did not complete within {self.max_wait_time} seconds"
        )

    def _create_documents_from_result(
        self,
        parsing_result: Dict[str, Any],
        source_path: Path,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Convert parsing results into LangChain Documents."""
        documents = []
        result_data = parsing_result.get("result", {})
        chunks = result_data.get("chunks", [])

        source_info = str(source_path) if self.store_full_path else source_path.name

        base_metadata = {
            "source": source_info,
            "file_name": source_path.name,
            "file_type": source_path.suffix.lower().lstrip("."),
            "parsing_job_id": parsing_result.get("id"),
            "chunking_strategy": result_data.get(
                "chunking_strategy", self.chunking_strategy
            ),
            "return_format": result_data.get("return_format", self.return_format),
            "element_types": result_data.get("element_types", self.element_types),
            "page_sizes": result_data.get("page_sizes"),
            "total_chunks": len(chunks),
        }

        if additional_metadata:
            base_metadata.update(additional_metadata)

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
                    set(elem.get("type") for elem in elements if elem.get("type"))
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
                    if len(pages) == 1:
                        chunk_metadata["page"] = pages[0]
                    else:
                        chunk_metadata["page_range"] = f"{min(pages)}-{max(pages)}"

            document = Document(page_content=chunk_content, metadata=chunk_metadata)
            documents.append(document)

        return documents

    def _process_single_file(
        self, file_path: Path, additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process a single file and return the documents."""
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            file_id = self._upload_file(file_path)

            job_id = self._create_parsing_job(file_id)

            parsing_result = self._wait_for_job_completion(job_id)

            documents = self._create_documents_from_result(
                parsing_result, file_path, additional_metadata
            )

            return documents

        except Exception as e:
            error_msg = f"Failed to parse {file_path}: {str(e)}"

            error_metadata = {
                "source": str(file_path) if self.store_full_path else file_path.name,
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower().lstrip("."),
                "parsing_error": error_msg,
                "parsing_status": "failed",
            }

            if additional_metadata:
                error_metadata.update(additional_metadata)

            return [Document(page_content="", metadata=error_metadata)]

    def lazy_load(self) -> Iterator[Document]:
        """Lazy loader that yields documents one by one."""
        for file_path in self.file_paths:
            documents = self._process_single_file(file_path)
            for doc in documents:
                yield doc

    def load(self) -> List[Document]:
        """Load all documents at once."""
        return list(self.lazy_load())

    async def alazy_load(self) -> Iterator[Document]:
        """Async lazy loader."""
        loop = asyncio.get_event_loop()

        for file_path in self.file_paths:
            documents = await loop.run_in_executor(
                None, self._process_single_file, file_path
            )
            for doc in documents:
                yield doc

    async def aload(self) -> List[Document]:
        """Async version of load."""
        documents = []
        async for doc in self.alazy_load():
            documents.append(doc)
        return documents
