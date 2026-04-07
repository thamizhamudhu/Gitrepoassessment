# batch_embed_chunks.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def batch_embed_with_chunks(text, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunks = splitter.split_text(text)

    emb = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vectors = emb.embed_documents(chunks)

    return {
        "num_chunks": len(chunks),
        "chunk_sample": chunks[:2],
        "embedding_dim": len(vectors[0]) if vectors else 0
    }


if __name__ == "__main__":
    sample_text = """
    LangChain is a framework designed to simplify the development of applications
    powered by large language models. It provides tools for chaining components,
    managing memory, and integrating with external data sources. One important
    concept in LangChain is text chunking, which helps in handling large documents
    by splitting them into smaller pieces before generating embeddings.
    """

    result = batch_embed_with_chunks(
        text=sample_text,
        chunk_size=100,
        overlap=20
    )

    print("Result:")
    print(result)