from mixedbread_ai_langchain import MixedbreadVectorStoreManager

manager = MixedbreadVectorStoreManager()

store_id = manager.create_vector_store(
    name="My Documents", description="Collection of research papers and notes"
)

print(f"Created vector store: {store_id}")

file_paths = ["./data/paper1.pdf", "./data/paper2.pdf", "./data/notes.docx"]
file_ids = manager.upload_files(store_id, file_paths)

print(f"Uploaded {len(file_ids)} files")
for i, file_id in enumerate(file_ids, 1):
    print(f"  {i}. File ID: {file_id}")

chunk_retriever = manager.create_retriever(
    vector_store_ids=[store_id], retriever_type="chunk", top_k=5
)

file_retriever = manager.create_retriever(
    vector_store_ids=[store_id], retriever_type="file", include_chunks=True
)

query = "machine learning techniques"

chunk_results = chunk_retriever.get_relevant_documents(query)
file_results = file_retriever.get_relevant_documents(query)

print(f"\nQuery: {query}")
print(f"Chunk retriever found: {len(chunk_results)} chunks")
print(f"File retriever found: {len(file_results)} files")

stores = manager.list_vector_stores()
print(f"\nTotal vector stores: {len(stores)}")

store_info = manager.get_vector_store_info(store_id)
print(f"Store name: {store_info.get('name')}")
print(f"Store description: {store_info.get('description')}")
