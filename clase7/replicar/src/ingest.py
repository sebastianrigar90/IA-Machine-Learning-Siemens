from pathlib import Path
import json

import faiss
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer


# =========================
# CONFIGURACIÓN
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documentos"
STORAGE_DIR = BASE_DIR / "storage"

INDEX_PATH = STORAGE_DIR / "index.faiss"
METADATA_PATH = STORAGE_DIR / "metadata.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# =========================
# EXTRACCIÓN DE PDF
# =========================

def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extrae el texto de cada página de un PDF."""

    pages = []

    document = fitz.open(pdf_path)

    for page_number, page in enumerate(document, start=1):
        text = page.get_text().strip()

        if text:
            pages.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": text,
                }
            )

    document.close()

    return pages


# =========================
# CHUNKING
# =========================

def chunk_text(text: str) -> list[str]:
    """Divide un texto en fragmentos con solapamiento."""

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def create_chunks(pages: list[dict]) -> list[dict]:
    """Convierte las páginas extraídas en chunks."""

    chunks = []

    for page_data in pages:
        text_chunks = chunk_text(page_data["text"])

        for chunk_number, text in enumerate(text_chunks, start=1):
            chunks.append(
                {
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunk": chunk_number,
                    "text": text,
                }
            )

    return chunks


# =========================
# INGESTA
# =========================

def ingest_documents():

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No se encontraron archivos PDF en: {DOCUMENTS_DIR}")
        return

    print(f"PDFs encontrados: {len(pdf_files)}")

    all_pages = []

    for pdf_file in pdf_files:
        print(f"Procesando: {pdf_file.name}")

        pages = extract_text_from_pdf(pdf_file)

        print(f"  Páginas con texto: {len(pages)}")

        all_pages.extend(pages)

    chunks = create_chunks(all_pages)

    print(f"Chunks generados: {len(chunks)}")

    if not chunks:
        print("No se generaron chunks.")
        return

    # -------------------------
    # EMBEDDINGS
    # -------------------------

    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]

    print("Generando embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    embeddings = embeddings.astype("float32")

    # -------------------------
    # FAISS
    # -------------------------

    print("Creando índice FAISS...")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    # -------------------------
    # GUARDAR
    # -------------------------

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nIngesta completada correctamente.")
    print(f"Índice: {INDEX_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Total vectores: {index.ntotal}")


if __name__ == "__main__":
    ingest_documents()