import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from ..common.client import MixedbreadClient


class MixedbreadVectorStoreManager:
    """
    Utility class for managing Mixedbread AI vector stores.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
    ):
        self._client = MixedbreadClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def create_vector_store(
        self, name: str, description: Optional[str] = None, **kwargs: Any
    ) -> str:
        try:
            response = self._client.client.vector_stores.create(
                name=name, description=description, **kwargs
            )
            return response.id
        except Exception as e:
            raise RuntimeError(f"Failed to create vector store: {str(e)}") from e

    async def acreate_vector_store(
        self, name: str, description: Optional[str] = None, **kwargs: Any
    ) -> str:
        try:
            response = await self._client.async_client.vector_stores.create(
                name=name, description=description, **kwargs
            )
            return response.id
        except Exception as e:
            raise RuntimeError(f"Failed to create vector store: {str(e)}") from e

    def upload_file(
        self, vector_store_id: str, file_path: Union[str, Path], **kwargs: Any
    ) -> str:
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(file_path, "rb") as f:
                response = self._client.client.vector_stores.files.upload_and_poll(
                    vector_store_id=vector_store_id, file=f, **kwargs
                )
            return response.id
        except Exception as e:
            raise RuntimeError(f"Failed to upload file {file_path}: {str(e)}") from e

    async def aupload_file(
        self, vector_store_id: str, file_path: Union[str, Path], **kwargs: Any
    ) -> str:
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            loop = asyncio.get_event_loop()

            def _upload_file():
                with open(file_path, "rb") as f:
                    return self._client.client.vector_stores.files.upload_and_poll(
                        vector_store_id=vector_store_id, file=f, **kwargs
                    )

            response = await loop.run_in_executor(None, _upload_file)
            return response.id
        except Exception as e:
            raise RuntimeError(f"Failed to upload file {file_path}: {str(e)}") from e

    def upload_files(
        self, vector_store_id: str, file_paths: List[Union[str, Path]], **kwargs: Any
    ) -> List[str]:
        file_ids = []
        for file_path in file_paths:
            try:
                file_id = self.upload_file(vector_store_id, file_path, **kwargs)
                file_ids.append(file_id)
            except Exception as e:
                print(f"Warning: Failed to upload {file_path}: {str(e)}")
                continue

        return file_ids

    async def aupload_files(
        self,
        vector_store_id: str,
        file_paths: List[Union[str, Path]],
        max_concurrent: int = 5,
        **kwargs: Any,
    ) -> List[str]:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def upload_single_file(file_path: Union[str, Path]) -> Optional[str]:
            async with semaphore:
                try:
                    return await self.aupload_file(vector_store_id, file_path, **kwargs)
                except Exception as e:
                    print(f"Warning: Failed to upload {file_path}: {str(e)}")
                    return None

        tasks = [upload_single_file(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        file_ids = [result for result in results if isinstance(result, str)]
        return file_ids

    def list_vector_stores(self, **kwargs: Any) -> List[Dict[str, Any]]:
        try:
            response = self._client.client.vector_stores.list(**kwargs)
            return [
                store.model_dump() if hasattr(store, "model_dump") else store
                for store in response.data
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to list vector stores: {str(e)}") from e

    async def alist_vector_stores(self, **kwargs: Any) -> List[Dict[str, Any]]:
        try:
            response = await self._client.async_client.vector_stores.list(**kwargs)
            return [
                store.model_dump() if hasattr(store, "model_dump") else store
                for store in response.data
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to list vector stores: {str(e)}") from e

    def get_vector_store_info(self, vector_store_id: str) -> Dict[str, Any]:
        try:
            response = self._client.client.vector_stores.retrieve(
                vector_store_id=vector_store_id
            )
            return (
                response.model_dump() if hasattr(response, "model_dump") else response
            )
        except Exception as e:
            raise RuntimeError(f"Failed to get vector store info: {str(e)}") from e

    async def aget_vector_store_info(self, vector_store_id: str) -> Dict[str, Any]:
        try:
            response = await self._client.async_client.vector_stores.retrieve(
                vector_store_id=vector_store_id
            )
            return (
                response.model_dump() if hasattr(response, "model_dump") else response
            )
        except Exception as e:
            raise RuntimeError(f"Failed to get vector store info: {str(e)}") from e

    def delete_vector_store(self, vector_store_id: str) -> bool:
        try:
            self._client.client.vector_stores.delete(vector_store_id=vector_store_id)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete vector store: {str(e)}") from e

    async def adelete_vector_store(self, vector_store_id: str) -> bool:
        try:
            await self._client.async_client.vector_stores.delete(
                vector_store_id=vector_store_id
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete vector store: {str(e)}") from e

    def create_retriever(
        self,
        vector_store_ids: List[str],
        retriever_type: str = "chunk",
        **retriever_kwargs: Any,
    ):
        if retriever_type == "chunk":
            from .vector_store_retriever import MixedbreadVectorStoreRetriever

            return MixedbreadVectorStoreRetriever(
                vector_store_ids=vector_store_ids,
                api_key=self._client.api_key,
                **retriever_kwargs,
            )
        elif retriever_type == "file":
            from .vector_store_file_retriever import MixedbreadVectorStoreFileRetriever

            return MixedbreadVectorStoreFileRetriever(
                vector_store_ids=vector_store_ids,
                api_key=self._client.api_key,
                **retriever_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown retriever type: {retriever_type}. Use 'chunk' or 'file'."
            )
