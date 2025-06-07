import pytest
import asyncio
from unittest.mock import Mock, patch
from mixedbread_ai_langchain import MixedbreadVectorStoreManager
from .test_config import TestConfig


class TestMixedbreadVectorStoreManager:
    def test_init_default(self, monkeypatch):
        """
        Test default initialization parameters for MixedbreadVectorStoreManager.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        manager = MixedbreadVectorStoreManager()
        
        assert manager._client is not None

    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for MixedbreadVectorStoreManager.
        """
        manager = MixedbreadVectorStoreManager(
            api_key="test-api-key",
            base_url="https://custom.api.com",
            timeout=30.0,
            max_retries=3,
        )
        
        assert manager._client is not None

    @patch("mixedbread_ai_langchain.retrievers.vector_store_manager.MixedbreadClient")
    def test_create_vector_store(self, mock_client_class):
        """
        Test creating a vector store.
        """
        # Mock the client and response
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.id = "vs_123"
        mock_client.client.vector_stores.create.return_value = mock_response
        
        manager = MixedbreadVectorStoreManager(api_key="fake-api-key")
        
        vector_store_id = manager.create_vector_store(
            name="Test Store",
            description="Test description",
            metadata={"project": "test"}
        )
        
        assert vector_store_id == "vs_123"
        mock_client.client.vector_stores.create.assert_called_once_with(
            name="Test Store",
            description="Test description", 
            metadata={"project": "test"}
        )

    @patch("mixedbread_ai_langchain.retrievers.vector_store_manager.MixedbreadClient")
    def test_update_vector_store(self, mock_client_class):
        """
        Test updating a vector store.
        """
        # Mock the client and response
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.model_dump.return_value = {"id": "vs_123", "name": "Updated Store"}
        mock_client.client.vector_stores.update.return_value = mock_response
        
        manager = MixedbreadVectorStoreManager(api_key="fake-api-key")
        
        result = manager.update_vector_store(
            vector_store_id="vs_123",
            name="Updated Store"
        )
        
        assert result == {"id": "vs_123", "name": "Updated Store"}
        mock_client.client.vector_stores.update.assert_called_once_with(
            vector_store_id="vs_123",
            name="Updated Store"
        )

    @patch("mixedbread_ai_langchain.retrievers.vector_store_manager.MixedbreadClient")
    def test_delete_vector_store(self, mock_client_class):
        """
        Test deleting a vector store.
        """
        # Mock the client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        manager = MixedbreadVectorStoreManager(api_key="fake-api-key")
        
        result = manager.delete_vector_store("vs_123")
        
        assert result is True
        mock_client.client.vector_stores.delete.assert_called_once_with(
            vector_store_id="vs_123"
        )

    @patch("mixedbread_ai_langchain.retrievers.vector_store_manager.MixedbreadClient")
    def test_list_vector_stores(self, mock_client_class):
        """
        Test listing vector stores.
        """
        # Mock the client and response
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_store = Mock()
        mock_store.model_dump.return_value = {"id": "vs_123", "name": "Test Store"}
        mock_response = Mock()
        mock_response.data = [mock_store]
        mock_client.client.vector_stores.list.return_value = mock_response
        
        manager = MixedbreadVectorStoreManager(api_key="fake-api-key")
        
        stores = manager.list_vector_stores(limit=10, offset=0)
        
        assert len(stores) == 1
        assert stores[0] == {"id": "vs_123", "name": "Test Store"}
        mock_client.client.vector_stores.list.assert_called_once_with(
            limit=10, offset=0
        )

    @patch("mixedbread_ai_langchain.retrievers.vector_store_manager.MixedbreadClient")
    def test_add_file_to_vector_store(self, mock_client_class):
        """
        Test adding a file to a vector store.
        """
        # Mock the client and response
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.model_dump.return_value = {"id": "file_123", "status": "completed"}
        mock_client.client.vector_stores.files.create.return_value = mock_response
        
        manager = MixedbreadVectorStoreManager(api_key="fake-api-key")
        
        result = manager.add_file_to_vector_store(
            vector_store_id="vs_123",
            file_id="file_123"
        )
        
        assert result == {"id": "file_123", "status": "completed"}
        mock_client.client.vector_stores.files.create.assert_called_once_with(
            vector_store_id="vs_123",
            file_id="file_123"
        )

    @patch("mixedbread_ai_langchain.retrievers.vector_store_manager.MixedbreadClient")
    def test_list_vector_store_files(self, mock_client_class):
        """
        Test listing files in a vector store.
        """
        # Mock the client and response
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_file = Mock()
        mock_file.model_dump.return_value = {"id": "file_123", "filename": "test.pdf"}
        mock_response = Mock()
        mock_response.data = [mock_file]
        mock_client.client.vector_stores.files.list.return_value = mock_response
        
        manager = MixedbreadVectorStoreManager(api_key="fake-api-key")
        
        files = manager.list_vector_store_files(
            vector_store_id="vs_123",
            limit=10
        )
        
        assert len(files) == 1
        assert files[0] == {"id": "file_123", "filename": "test.pdf"}
        mock_client.client.vector_stores.files.list.assert_called_once_with(
            vector_store_id="vs_123",
            limit=10
        )

    @patch("mixedbread_ai_langchain.retrievers.vector_store_manager.MixedbreadClient")
    def test_delete_vector_store_file(self, mock_client_class):
        """
        Test deleting a file from a vector store.
        """
        # Mock the client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        manager = MixedbreadVectorStoreManager(api_key="fake-api-key")
        
        result = manager.delete_vector_store_file(
            vector_store_id="vs_123",
            file_id="file_123"
        )
        
        assert result is True
        mock_client.client.vector_stores.files.delete.assert_called_once_with(
            vector_store_id="vs_123",
            file_id="file_123"
        )

    def test_create_retriever_chunk(self):
        """
        Test creating a chunk-based retriever.
        """
        manager = MixedbreadVectorStoreManager(api_key="fake-api-key")
        
        retriever = manager.create_retriever(
            vector_store_ids=["vs_123"],
            retriever_type="chunk",
            top_k=5
        )
        
        from mixedbread_ai_langchain import MixedbreadVectorStoreRetriever
        assert isinstance(retriever, MixedbreadVectorStoreRetriever)
        assert retriever.vector_store_ids == ["vs_123"]
        assert retriever.top_k == 5

    def test_create_retriever_file(self):
        """
        Test creating a file-based retriever.
        """
        manager = MixedbreadVectorStoreManager(api_key="fake-api-key")
        
        retriever = manager.create_retriever(
            vector_store_ids=["vs_123"],
            retriever_type="file",
            top_k=3
        )
        
        from mixedbread_ai_langchain import MixedbreadVectorStoreFileRetriever
        assert isinstance(retriever, MixedbreadVectorStoreFileRetriever)
        assert retriever.vector_store_ids == ["vs_123"]
        assert retriever.top_k == 3

    def test_create_retriever_invalid_type(self):
        """
        Test creating a retriever with invalid type.
        """
        manager = MixedbreadVectorStoreManager(api_key="fake-api-key")
        
        with pytest.raises(ValueError, match="Unknown retriever type"):
            manager.create_retriever(
                vector_store_ids=["vs_123"],
                retriever_type="invalid"
            )

    @pytest.mark.asyncio
    @patch("mixedbread_ai_langchain.retrievers.vector_store_manager.AsyncMixedbread")
    async def test_direct_async_client_access(self, mock_async_client_class):
        """
        Test direct async client access for vector store operations.
        """
        # Mock the async client and response
        mock_async_client = Mock()
        mock_async_client_class.return_value = mock_async_client
        
        mock_response = Mock()
        mock_response.id = "vs_async_123"
        mock_async_client.vector_stores.create.return_value = mock_response
        
        manager = MixedbreadVectorStoreManager(api_key="fake-api-key")
        
        # Test direct async client access
        response = await manager.aclient.vector_stores.create(
            name="Direct Async Test Store"
        )
        
        assert response.id == "vs_async_123"
        mock_async_client.vector_stores.create.assert_called_once_with(
            name="Direct Async Test Store"
        )

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_vector_store_operations(self):
        """
        Test basic vector store operations with real API calls.
        Note: This test creates and deletes a real vector store.
        """
        manager = MixedbreadVectorStoreManager(**TestConfig.get_test_embedder_config())
        
        # Create a test vector store using sync convenience method
        vector_store_id = manager.create_vector_store(
            name="Test Integration Store",
            description="Created by integration test"
        )
        
        assert isinstance(vector_store_id, str)
        assert len(vector_store_id) > 0
        
        try:
            # Get vector store info using sync convenience method
            info = manager.get_vector_store_info(vector_store_id)
            assert info["id"] == vector_store_id
            assert info["name"] == "Test Integration Store"
            
            # Test direct client access
            direct_info = manager.client.vector_stores.retrieve(vector_store_id)
            assert direct_info.id == vector_store_id
            assert direct_info.name == "Test Integration Store"
            
            # List vector stores (should include our test store)
            stores = manager.list_vector_stores(limit=50)
            assert isinstance(stores, list)
            store_ids = [store["id"] for store in stores]
            assert vector_store_id in store_ids
            
            # Test direct client list
            direct_stores = manager.client.vector_stores.list(limit=50)
            direct_store_ids = [store.id for store in direct_stores.data]
            assert vector_store_id in direct_store_ids
            
        finally:
            # Clean up: delete the test vector store
            result = manager.delete_vector_store(vector_store_id)
            assert result is True
            
    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration  
    @pytest.mark.asyncio
    async def test_integration_async_vector_store_operations(self):
        """
        Test async vector store operations using direct client access.
        Note: This test creates and deletes a real vector store asynchronously.
        """
        manager = MixedbreadVectorStoreManager(**TestConfig.get_test_embedder_config())
        
        if not manager.aclient:
            pytest.skip("AsyncMixedbread not available")
        
        # Create a test vector store using direct async client
        response = await manager.aclient.vector_stores.create(
            name="Test Async Integration Store",
            description="Created by async integration test"
        )
        vector_store_id = response.id
        
        assert isinstance(vector_store_id, str)
        assert len(vector_store_id) > 0
        
        try:
            # Get vector store info using direct async client
            info = await manager.aclient.vector_stores.retrieve(vector_store_id)
            assert info.id == vector_store_id
            assert info.name == "Test Async Integration Store"
            
            # List vector stores using direct async client
            stores_response = await manager.aclient.vector_stores.list(limit=50)
            store_ids = [store.id for store in stores_response.data]
            assert vector_store_id in store_ids
            
        finally:
            # Clean up: delete the test vector store using direct async client
            await manager.aclient.vector_stores.delete(vector_store_id)
            # Note: The async delete doesn't return a boolean, just completes successfully