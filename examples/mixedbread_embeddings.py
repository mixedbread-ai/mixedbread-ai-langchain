from mixedbread_ai_langchain import MixedbreadEmbeddings

embeddings = MixedbreadEmbeddings()

text = "German bread is known for its variety and quality"
embedding = embeddings.embed_query(text)

print("Example 1 - Single Text:")
print(f"Model: {embeddings.model}")
print(f"Text: {text}")
print(f"Vector dimension: {len(embedding)}")
print(f"Vector preview: {embedding[:3]}...{embedding[-3:]}")

texts = [
    "Rye bread is a staple in German cuisine",
    "Brezel (pretzel) is a popular German snack",
    "Pumpernickel is a dark, dense German bread",
]

doc_embeddings = embeddings.embed_documents(texts)

print("\nExample 2 - Multiple Texts:")
print(f"Model: {embeddings.model}")
print(f"Number of texts: {len(texts)}")
for i, (text, embedding) in enumerate(zip(texts, doc_embeddings)):
    print(f"\nText {i+1}: {text}")
    print(f"Vector dimension: {len(embedding)}")
    print(f"Vector preview: {embedding[:3]}...{embedding[-3:]}")
