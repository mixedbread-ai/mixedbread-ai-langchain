from langchain_core.documents import Document
from mixedbread_ai_langchain import MixedbreadReranker

documents = [
    Document(
        page_content="German rye bread is known for its dense texture and rich flavor"
    ),
    Document(
        page_content="Pretzels are a traditional German baked good often served with beer"
    ),
    Document(
        page_content="Black Forest cake is a famous German dessert made with chocolate and cherries"
    ),
    Document(
        page_content="Stollen is a traditional German Christmas bread with dried fruits"
    ),
    Document(
        page_content="German bakeries are famous for their sourdough bread recipes"
    ),
]

# Initialize the reranker
reranker = MixedbreadReranker(model="mixedbread-ai/mxbai-rerank-large-v2", top_k=3)

query = "Tell me about German bread traditions"

# Example 1: Sync reranking
print("=== Sync reranking ===")
reranked_docs = reranker.compress_documents(documents, query)

print(f"Query: {query}")
print(f"Reranked top {len(reranked_docs)} documents:")

for i, doc in enumerate(reranked_docs, 1):
    score = doc.metadata.get("rerank_score", 0)
    original_index = doc.metadata.get("original_index", "unknown")
    print(
        f"{i}. Score: {score:.3f} (original index: {original_index}) - {doc.page_content}"
    )

print("\n" + "=" * 50 + "\n")
