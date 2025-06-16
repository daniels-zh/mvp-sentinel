import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Cargar índice FAISS y metadatos
index = faiss.read_index("vector_index.faiss")
with open("metadatos_fragments.json", "r", encoding="utf-8") as f:
    metadatos = json.load(f)

# Cargar modelo de embeddings
embedder = SentenceTransformer("all-mpnet-base-v2")
#embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Configuración de la app
st.set_page_config(page_title="MVP Sentinel", layout="wide")
st.title("🛠️ MVP Sentinel - Consulta Inteligente de Documentos")

# Interfaz de usuario
query = st.text_input("Ingresa tu consulta sobre la documentación:")

if query:
    with st.spinner("Procesando consulta..."):
        # Generar embedding de la consulta
        query_embedding = embedder.encode([query])[0].astype("float32")

        # Buscar los k fragmentos más cercanos
        k = 5
        D, I = index.search(np.array([query_embedding]), k)

        # Recuperar textos más relevantes
        contexto = "\n".join([metadatos[str(i)]["texto"] for i in I[0]])

        # Enviar a OpenAI con el contexto
        prompt = f"Responde en base al siguiente contexto:\n{contexto}\n\nPregunta: {query}\nRespuesta:"

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un experto en minería y normativa industrial."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )

        st.markdown("### 📄 Respuesta:")
        st.write(response["choices"][0]["message"]["content"])