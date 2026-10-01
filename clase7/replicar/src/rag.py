import os

from dotenv import load_dotenv
from openai import OpenAI

from retrieval import search


# =========================
# CARGAR VARIABLES DE ENTORNO
# =========================

load_dotenv()


# =========================
# CONFIGURACIÓN AZURE OPENAI
# =========================

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


if not AZURE_OPENAI_ENDPOINT:
    raise ValueError("Falta AZURE_OPENAI_ENDPOINT en el archivo .env")

if not AZURE_OPENAI_API_KEY:
    raise ValueError("Falta AZURE_OPENAI_API_KEY en el archivo .env")

if not AZURE_OPENAI_DEPLOYMENT:
    raise ValueError("Falta AZURE_OPENAI_DEPLOYMENT en el archivo .env")


# =========================
# CLIENTE AZURE OPENAI
# =========================

client = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/"
)


# =========================
# CONSTRUIR CONTEXTO
# =========================

def build_context(results):

    context_parts = []

    for i, result in enumerate(results, start=1):

        context_parts.append(
            f"""
[DOCUMENTO {i}]
Fuente: {result['source']}
Página: {result['page']}

{result['text']}
"""
        )

    return "\n\n".join(context_parts)


# =========================
# CONSULTAR EL RAG
# =========================

def ask(question, k=3):

    print("\nBuscando información en FAISS local...")

    results = search(question, top_k=k)

    context = build_context(results)

    print("\nConsultando Azure OpenAI...")

    response = client.responses.create(

        model=AZURE_OPENAI_DEPLOYMENT,

        instructions="""
You are a technical assistant specialized in electrical transformers.

Answer the user's question using ONLY the information provided
in the retrieved context.

If the retrieved context does not contain enough information,
say clearly that the information is not available in the provided
documentation.

Do not invent information.

Provide a concise but useful technical answer.
""",

        input=f"""
Retrieved context:

{context}

User question:

{question}
"""
    )

    return response.output_text, results


# =========================
# EJECUCIÓN PRINCIPAL
# =========================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("RAG LOCAL PARA DOCUMENTACIÓN DE TRANSFORMADORES")
    print("=" * 60)

    question = input(
        "\nPregunta sobre el transformador: "
    )

    answer, sources = ask(question)

    print("\n" + "=" * 70)
    print("RESPUESTA")
    print("=" * 70)

    print(answer)

    print("\n" + "=" * 70)
    print("FUENTES RECUPERADAS")
    print("=" * 70)

    for i, source in enumerate(sources, start=1):

        print(
            f"{i}. {source['source']} "
            f"- Página {source['page']} "
            f"- Score: {source['score']:.4f}"
        )