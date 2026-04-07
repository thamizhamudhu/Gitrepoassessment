# compare_embeddings.py

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def compare_embedding_models(sentence):
    emb = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vec = emb.embed_query(sentence)

    return {
        "model": "gemini",
        "dimension": len(vec)
    }


if __name__ == "__main__":
    sample_sentence = "LangChain helps in building LLM powered applications."

    result = compare_embedding_models(sample_sentence)

    print("Result:")
    print(result)