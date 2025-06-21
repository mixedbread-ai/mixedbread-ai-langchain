from mixedbread_ai_langchain import MixedbreadDocumentLoader


def basic_usage():
    """Basic document loading usage."""
    print("=== Basic Document Loading ===")

    # Initialize document loader
    loader = MixedbreadDocumentLoader(
        file_path="data/acme_invoice.pdf",  # Path to your document
        # api_key="your-api-key"  # or set MXBAI_API_KEY env var
    )

    try:
        # Load and parse the document
        documents = loader.load()

        print(f"Loaded {len(documents)} document chunks")
        print()

        for i, doc in enumerate(documents):
            print(f"Chunk {i + 1}:")
            print(f"  Content: {doc.page_content[:100]}...")
            print(f"  Source: {doc.metadata.get('source', 'unknown')}")
            print(f"  Pages: {doc.metadata.get('pages', 'N/A')}")
            print(f"  Chunk Index: {doc.metadata.get('chunk_index', 'N/A')}")
            print(f"  Total Chunks: {doc.metadata.get('total_chunks', 'N/A')}")
            print()

            if i >= 2:  # Show only first 3 chunks
                print("...")
                break

    except Exception as e:
        print(f"Error loading document: {e}")
    print()


def custom_configuration():
    """Example with custom parsing configuration."""
    print("=== Custom Configuration ===")

    # Initialize with custom settings
    loader = MixedbreadDocumentLoader(
        file_path="data/acme_invoice.pdf",
        chunking_strategy="page",  # Chunk by page
        return_format="plain",  # Plain text instead of markdown
        element_types=["text", "title"],  # Only extract text and titles
        max_wait_time=600,  # Wait up to 10 minutes
        poll_interval=10,  # Poll every 10 seconds
    )

    try:
        documents = loader.load()

        print(f"Custom parsing produced {len(documents)} chunks")

        if documents:
            sample_doc = documents[0]
            print(f"Sample content: {sample_doc.page_content[:150]}...")
            print(f"Parsing job ID: {sample_doc.metadata.get('parsing_job_id', 'N/A')}")

    except Exception as e:
        print(f"Error with custom configuration: {e}")
    print()


def main():
    """Run all examples."""
    # Note: Set MXBAI_API_KEY environment variable before running
    # Make sure you have a test PDF file at data/acme_invoice.pdf

    print("Mixedbread Document Loader Examples")
    print("=" * 40)

    try:
        basic_usage()
        custom_configuration()

    except Exception as e:
        print(f"Example error: {e}")
        print("Make sure to:")
        print("1. Set MXBAI_API_KEY environment variable")
        print("2. Have a test PDF file at data/acme_invoice.pdf")


if __name__ == "__main__":
    main()
