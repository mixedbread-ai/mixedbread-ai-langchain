import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from langchain_core.utils import Secret
from ..common.client import MixedbreadClient
from ..common.mixins import SerializationMixin, AsyncMixin, ErrorHandlingMixin
from ..common.logging import get_logger

logger = get_logger(__name__)

# Import async client directly from mixedbread SDK
try:
    from mixedbread import AsyncMixedbread
except ImportError:
    # Fallback for older SDK versions
    AsyncMixedbread = None


class MixedbreadVectorStoreManager(SerializationMixin, AsyncMixin, ErrorHandlingMixin):
    """
    Comprehensive manager for Mixedbread AI vector stores.
    
    Provides full CRUD operations for vector stores and files. Uses direct
    access to sync and async clients to eliminate method duplication.
    
    Usage:
        # Sync operations
        manager = MixedbreadVectorStoreManager()
        store_id = manager.create_vector_store("My Store")
        
        # Async operations (direct client access)
        store_id = (await manager.aclient.vector_stores.create(name="My Store")).id
    """

    def __init__(
        self,
        api_key: Union[Secret, str, None] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
    ):
        """
        Initialize the vector store manager.
        
        Args:
            api_key: API key for Mixedbread AI (or set MXBAI_API_KEY env var).
            base_url: Base URL for the API.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
        """
        # Initialize sync client for convenience methods
        self._client = MixedbreadClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        
        # Initialize async client for direct async access
        if AsyncMixedbread:
            resolved_api_key = api_key
            if isinstance(api_key, Secret):
                resolved_api_key = api_key.resolve_value()
            elif api_key is None:
                import os
                resolved_api_key = os.environ.get("MXBAI_API_KEY")
                
            self.aclient = AsyncMixedbread(
                api_key=resolved_api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            logger.warning("AsyncMixedbread not available. Async operations will use fallback method.")
            self.aclient = None
            
        # Expose sync client for advanced usage
        self.client = self._client.client

    def create_vector_store(
        self, 
        name: str, 
        description: Optional[str] = None,
        expires_after: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        file_ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """
        Create a new vector store.
        
        Args:
            name: Name of the vector store.
            description: Optional description.
            expires_after: Expiration policy (e.g., {"anchor": "last_active_at", "days": 7}).
            metadata: Optional metadata dictionary.
            file_ids: Optional list of file IDs to add to the store.
            **kwargs: Additional parameters.
            
        Returns:
            The ID of the created vector store.
        """
        try:
            logger.info(f"Creating vector store: {name}")
            
            create_params = {"name": name}
            if description:
                create_params["description"] = description
            if expires_after:
                create_params["expires_after"] = expires_after
            if metadata:
                create_params["metadata"] = metadata
            if file_ids:
                create_params["file_ids"] = file_ids
                
            create_params.update(kwargs)
            
            response = self._client.client.vector_stores.create(**create_params)
            logger.info(f"Successfully created vector store {response.id}")
            return response.id
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise RuntimeError(f"Failed to create vector store: {str(e)}") from e

    # Note: For async operations, use manager.aclient.vector_stores.create() directly

    def update_vector_store(
        self, 
        vector_store_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        expires_after: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Update an existing vector store.
        
        Args:
            vector_store_id: ID of the vector store to update.
            name: New name for the vector store.
            description: New description.
            expires_after: New expiration policy.
            metadata: New metadata dictionary.
            **kwargs: Additional parameters.
            
        Returns:
            Updated vector store information.
        """
        try:
            logger.info(f"Updating vector store: {vector_store_id}")
            
            update_params = {}
            if name is not None:
                update_params["name"] = name
            if description is not None:
                update_params["description"] = description
            if expires_after is not None:
                update_params["expires_after"] = expires_after
            if metadata is not None:
                update_params["metadata"] = metadata
                
            update_params.update(kwargs)
            
            response = self._client.client.vector_stores.update(
                vector_store_id=vector_store_id, **update_params
            )
            logger.info(f"Successfully updated vector store {vector_store_id}")
            return response.model_dump() if hasattr(response, "model_dump") else response
        except Exception as e:
            logger.error(f"Failed to update vector store: {str(e)}")
            raise RuntimeError(f"Failed to update vector store: {str(e)}") from e
            
    # Note: For async operations, use manager.aclient.vector_stores.update() directly

    def add_file_to_vector_store(
        self, vector_store_id: str, file_id: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Add an existing file to a vector store.
        
        Args:
            vector_store_id: ID of the vector store.
            file_id: ID of the file to add.
            **kwargs: Additional parameters.
            
        Returns:
            Vector store file information.
        """
        try:
            logger.info(f"Adding file {file_id} to vector store {vector_store_id}")
            
            response = self._client.client.vector_stores.files.create(
                vector_store_id=vector_store_id, file_id=file_id, **kwargs
            )
            logger.info(f"Successfully added file to vector store")
            return response.model_dump() if hasattr(response, "model_dump") else response
        except Exception as e:
            logger.error(f"Failed to add file to vector store: {str(e)}")
            raise RuntimeError(f"Failed to add file to vector store: {str(e)}") from e
            
    # Note: For async operations, use manager.aclient.vector_stores.files.create() directly
    
    def upload_file(
        self, vector_store_id: str, file_path: Union[str, Path], **kwargs: Any
    ) -> str:
        """
        Upload a file and add it to a vector store.
        
        Args:
            vector_store_id: ID of the vector store.
            file_path: Path to the file to upload.
            **kwargs: Additional parameters.
            
        Returns:
            The file ID of the uploaded file.
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            logger.info(f"Uploading file {file_path} to vector store {vector_store_id}")
            
            with open(file_path, "rb") as f:
                response = self._client.client.vector_stores.files.upload_and_poll(
                    vector_store_id=vector_store_id, file=f, **kwargs
                )
            logger.info(f"Successfully uploaded file to vector store")
            return response.id
        except Exception as e:
            logger.error(f"Failed to upload file {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to upload file {file_path}: {str(e)}") from e

    # Note: For async file uploads, use manager.aclient.vector_stores.files.upload_and_poll() directly

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

    # Note: For async bulk uploads, implement using manager.aclient directly with asyncio.gather()

    def list_vector_stores(
        self, limit: Optional[int] = 20, offset: Optional[int] = 0, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        List vector stores with pagination support.
        
        Args:
            limit: Maximum number of vector stores to return.
            offset: Number of vector stores to skip.
            **kwargs: Additional parameters.
            
        Returns:
            List of vector store information dictionaries.
        """
        try:
            logger.info(f"Listing vector stores (limit={limit}, offset={offset})")
            
            list_params = {}
            if limit is not None:
                list_params["limit"] = limit
            if offset is not None:
                list_params["offset"] = offset
            list_params.update(kwargs)
            
            response = self._client.client.vector_stores.list(**list_params)
            stores = [
                store.model_dump() if hasattr(store, "model_dump") else store
                for store in response.data
            ]
            logger.info(f"Retrieved {len(stores)} vector stores")
            return stores
        except Exception as e:
            logger.error(f"Failed to list vector stores: {str(e)}")
            raise RuntimeError(f"Failed to list vector stores: {str(e)}") from e

    # Note: For async operations, use manager.aclient.vector_stores.list() directly

    def get_vector_store_info(self, vector_store_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a vector store.
        
        Args:
            vector_store_id: ID of the vector store.
            
        Returns:
            Vector store information dictionary.
        """
        try:
            logger.info(f"Getting vector store info: {vector_store_id}")
            
            response = self._client.client.vector_stores.retrieve(
                vector_store_id=vector_store_id
            )
            info = response.model_dump() if hasattr(response, "model_dump") else response
            logger.info(f"Retrieved vector store info")
            return info
        except Exception as e:
            logger.error(f"Failed to get vector store info: {str(e)}")
            raise RuntimeError(f"Failed to get vector store info: {str(e)}") from e

    # Note: For async operations, use manager.aclient.vector_stores.retrieve() directly

    def delete_vector_store(self, vector_store_id: str) -> bool:
        """
        Delete a vector store.
        
        Args:
            vector_store_id: ID of the vector store to delete.
            
        Returns:
            True if deletion was successful.
        """
        try:
            logger.info(f"Deleting vector store: {vector_store_id}")
            
            self._client.client.vector_stores.delete(vector_store_id=vector_store_id)
            logger.info(f"Successfully deleted vector store {vector_store_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector store: {str(e)}")
            raise RuntimeError(f"Failed to delete vector store: {str(e)}") from e

    # Note: For async operations, use manager.aclient.vector_stores.delete() directly

    # Vector Store File Management Methods
    
    def list_vector_store_files(
        self, 
        vector_store_id: str, 
        limit: Optional[int] = 20, 
        offset: Optional[int] = 0, 
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        List files in a vector store.
        
        Args:
            vector_store_id: ID of the vector store.
            limit: Maximum number of files to return.
            offset: Number of files to skip.
            **kwargs: Additional parameters.
            
        Returns:
            List of file information dictionaries.
        """
        try:
            logger.info(f"Listing files in vector store {vector_store_id}")
            
            list_params = {}
            if limit is not None:
                list_params["limit"] = limit
            if offset is not None:
                list_params["offset"] = offset
            list_params.update(kwargs)
            
            response = self._client.client.vector_stores.files.list(
                vector_store_id=vector_store_id, **list_params
            )
            files = [
                file.model_dump() if hasattr(file, "model_dump") else file
                for file in response.data
            ]
            logger.info(f"Retrieved {len(files)} files from vector store")
            return files
        except Exception as e:
            logger.error(f"Failed to list vector store files: {str(e)}")
            raise RuntimeError(f"Failed to list vector store files: {str(e)}") from e
            
    # Note: For async operations, use manager.aclient.vector_stores.files.list() directly
            
    def get_vector_store_file_info(
        self, vector_store_id: str, file_id: str
    ) -> Dict[str, Any]:
        """
        Get information about a file in a vector store.
        
        Args:
            vector_store_id: ID of the vector store.
            file_id: ID of the file.
            
        Returns:
            File information dictionary.
        """
        try:
            logger.info(f"Getting file info {file_id} from vector store {vector_store_id}")
            
            response = self._client.client.vector_stores.files.retrieve(
                vector_store_id=vector_store_id, file_id=file_id
            )
            info = response.model_dump() if hasattr(response, "model_dump") else response
            logger.info(f"Retrieved file info")
            return info
        except Exception as e:
            logger.error(f"Failed to get vector store file info: {str(e)}")
            raise RuntimeError(f"Failed to get vector store file info: {str(e)}") from e
            
    # Note: For async operations, use manager.aclient.vector_stores.files.retrieve() directly
            
    def delete_vector_store_file(
        self, vector_store_id: str, file_id: str
    ) -> bool:
        """
        Delete a file from a vector store.
        
        Args:
            vector_store_id: ID of the vector store.
            file_id: ID of the file to delete.
            
        Returns:
            True if deletion was successful.
        """
        try:
            logger.info(f"Deleting file {file_id} from vector store {vector_store_id}")
            
            self._client.client.vector_stores.files.delete(
                vector_store_id=vector_store_id, file_id=file_id
            )
            logger.info(f"Successfully deleted file from vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector store file: {str(e)}")
            raise RuntimeError(f"Failed to delete vector store file: {str(e)}") from e
            
    # Note: For async operations, use manager.aclient.vector_stores.files.delete() directly
    
    # Retriever Factory Methods
    
    def create_retriever(
        self,
        vector_store_ids: List[str],
        retriever_type: str = "chunk",
        **retriever_kwargs: Any,
    ):
        """
        Create a retriever for the vector stores with both sync and async client access.
        
        Args:
            vector_store_ids: List of vector store IDs to search.
            retriever_type: Type of retriever ("chunk" or "file").
            **retriever_kwargs: Additional arguments for the retriever.
            
        Returns:
            Configured retriever instance with both sync and async capabilities.
        """
        if retriever_type == "chunk":
            from .vector_store_retriever import MixedbreadVectorStoreRetriever

            return MixedbreadVectorStoreRetriever(
                vector_store_ids=vector_store_ids,
                sync_client=self._client,
                async_client=self.aclient,
                **retriever_kwargs,
            )
        elif retriever_type == "file":
            from .vector_store_file_retriever import MixedbreadVectorStoreFileRetriever

            return MixedbreadVectorStoreFileRetriever(
                vector_store_ids=vector_store_ids,
                sync_client=self._client,
                async_client=self.aclient,
                **retriever_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown retriever type: {retriever_type}. Use 'chunk' or 'file'."
            )
