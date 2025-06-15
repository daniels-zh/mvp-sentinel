import faiss
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv
import os

# Cargar API Key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cargar metadatos
with open("metadatos_fragments.json", "r", encoding="utf-8") as f:
    metadatos = json.load(f)

# Detectar la clave correcta del texto
clave_texto = "texto" if "texto" in metadatos[0] else "fragmento"
fragments = [item[clave_texto] for item in metadatos]
embeddings = np.array([item["embedding"] for item in metadatos], dtype=np.float32)

# Cargar índice FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Función para generar embedding con OpenAI
def generar_embedding_openai(texto):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return response.data[0].embedding

# Bucle interactivo
print("\n💬 Prueba el motor de búsqueda semántica.\n")
while True:
    consulta = input("Escribe una consulta (o 'salir' para terminar): ")
    if consulta.lower() == "salir":
        break

    query_vec = generar_embedding_openai(consulta)
    D, I = index.search(np.array([query_vec], dtype=np.float32), k=5)

    print("\n🔎 Resultados más relevantes:\n")
    for i, idx in enumerate(I[0]):
        score = 1 - D[0][i] / 4  # Similitud aproximada
        print(f"🔹 Fragmento {i+1} (Score aprox: {score:.2f}):\n{fragments[idx][:500]}\n{'-'*80}")