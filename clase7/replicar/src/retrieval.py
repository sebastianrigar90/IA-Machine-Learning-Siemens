from pathlib import Path
import json

import faiss
from sentence_transformers import SentenceTransformer


# =========================
# CONFIGURACIÓN
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

STORAGE_DIR = BASE_DIR / "storage"

INDEX_PATH = STORAGE_DIR / "index.faiss"
METADATA_PATH = STORAGE_DIR / "metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# =========================
# CARGAR RECURSOS
# =========================

print("Cargando modelo de embeddings...")
model = SentenceTransformer(EMBEDDING_MODEL)

print("Cargando índice FAISS...")
index = faiss.read_index(str(INDEX_PATH))

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)


# =========================
# BÚSQUEDA
# =========================

def search(query: str, top_k: int = 3) -> list[dict]:
    """
    Busca los fragmentos más relevantes para una consulta.
    """

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "score": float(distance),
            "text": metadata[idx]["text"],
            "source": metadata[idx]["source"],
            "page": metadata[idx]["page"],
        })

    return results


# =========================
# PRUEBA
# =========================

if __name__ == "__main__":

    question = "What are the maintenance requirements for the transformer?"

    print(f"\nPregunta: {question}\n")

    results = search(question)

    for position, result in enumerate(results, start=1):

        print(f"--- Resultado {position} ---")
        print(f"Fuente: {result['source']}")
        print(f"Página: {result['page']}")
        print(f"Score: {result['score']:.4f}")
        print()
        print(result["text"])
        print("\n")