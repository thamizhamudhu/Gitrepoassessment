import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def cosine_similarity_manual(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    return dot / (norm1 * norm2)


def cosine_similarity_numpy(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def compare_word_pairs():
    emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    print("🔄 Generating embeddings...")

    v1 = emb.embed_query("cat")
    v2 = emb.embed_query("dog")
    v3 = emb.embed_query("car")

    print("📐 Calculating similarities...")

    sim_cat_dog_np = cosine_similarity_numpy(v1, v2)
    sim_cat_car_np = cosine_similarity_numpy(v1, v3)

    sim_cat_dog_manual = cosine_similarity_manual(v1, v2)
    sim_cat_car_manual = cosine_similarity_manual(v1, v3)

    return {
        "numpy": {
            "cat-dog": sim_cat_dog_np,
            "cat-car": sim_cat_car_np,
        },
        "manual": {
            "cat-dog": sim_cat_dog_manual,
            "cat-car": sim_cat_car_manual,
        },
        "more_similar": "cat-dog" if sim_cat_dog_np > sim_cat_car_np else "cat-car"
    }


if __name__ == "__main__":
    result = compare_word_pairs()

    print("\n🔹 Similarity Results (NumPy):")
    print(f"cat ↔ dog : {result['numpy']['cat-dog']:.4f}")
    print(f"cat ↔ car : {result['numpy']['cat-car']:.4f}")

    print("\n🔹 Similarity Results (Manual):")
    print(f"cat ↔ dog : {result['manual']['cat-dog']:.4f}")
    print(f"cat ↔ car : {result['manual']['cat-car']:.4f}")

    print("\n🏆 Most Similar Pair:")
    print(result["more_similar"])