from langchain_core.documents import Document
from mixedbread_ai_langchain import MixedbreadEmbeddings

embeddings = MixedbreadEmbeddings()

query = "What is German bread like?"
query_embedding = embeddings.embed_query(query)

print(f"Query: {query}")
print(f"Embedding dimension: {len(query_embedding)}")
print(f"Embedding vector: {query_embedding[:5]}...")

texts = [
    "German rye bread is known for its dense texture and rich flavor",
    "Pretzels are a traditional German baked good often served with beer",
    "Black Forest cake is a famous German dessert made with chocolate and cherries",
]

doc_embeddings = embeddings.embed_documents(texts)

print(f"\nEmbedded {len(doc_embeddings)} documents")
for i, embedding in enumerate(doc_embeddings):
    print(f"Document {i+1} embedding dimension: {len(embedding)}")

documents = [
    Document(content=text, metadata={"category": "food", "origin": "German"})
    for text in texts
]

documents_with_embeddings = embeddings.embed_langchain_documents(documents)

print(f"\nEmbedded {len(documents_with_embeddings)} LangChain Documents")
print(
    f"First document embedding: {documents_with_embeddings[0].metadata['embedding'][:3]}..."
)
