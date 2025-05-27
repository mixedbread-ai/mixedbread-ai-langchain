from mixedbread_ai_langchain import MixedbreadDocumentLoader

loader = MixedbreadDocumentLoader(
    file_paths=["./data/report.pdf"], chunking_strategy="page", return_format="markdown"
)

documents = loader.load()

print(f"Loaded {len(documents)} document chunks")
print(f"First chunk content: {documents[0].page_content[:100]}...")
print(f"Source: {documents[0].metadata['source']}")
print(f"File type: {documents[0].metadata['file_type']}")
print(f"Chunk index: {documents[0].metadata['chunk_index']}")
