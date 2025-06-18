import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configuraciones
RUTA_FRAGMENTOS = "context"
UMBRAL_CARACTERES = 250  # mínimo de caracteres para conservar un fragmento
NOMBRE_INDICE_FAISS = "vector_index.faiss"
NOMBRE_METADATOS = "metadatos_fragments.json"
MODELO_EMBEDDING = "all-mpnet-base-v2"

# Cargar modelo de embeddings
print("🔄 Cargando modelo all-mpnet-base-v2...")
modelo = SentenceTransformer(MODELO_EMBEDDING)

# Leer todos los archivos .txt desde el directorio de fragmentos
fragmentos = []
for archivo in os.listdir(RUTA_FRAGMENTOS):
    if archivo.endswith(".txt"):
        ruta_completa = os.path.join(RUTA_FRAGMENTOS, archivo)
        with open(ruta_completa, "r", encoding="utf-8") as f:
            texto = f.read()
            fragmentos.append({
                "archivo": archivo,
                "texto": texto.strip()
            })

# Filtrar por longitud
fragmentos_filtrados = []
for frag in fragmentos:
    if len(frag["texto"]) >= UMBRAL_CARACTERES:
        fragmentos_filtrados.append(frag)
print(f"🧹 Fragmentos conservados tras filtrado: {len(fragmentos_filtrados)} / {len(fragmentos)}")

# Generar embeddings
print("🔢 Generando embeddings...")
textos = [f["texto"] for f in fragmentos_filtrados]
embeddings = modelo.encode(textos, show_progress_bar=True)

# Guardar en índice FAISS
print("📦 Creando índice FAISS...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype=np.float32))
faiss.write_index(index, NOMBRE_INDICE_FAISS)
print(f"✅ Índice FAISS guardado en: {NOMBRE_INDICE_FAISS}")

# Guardar metadatos
metadatos = []
for i, frag in enumerate(fragmentos_filtrados):
    metadatos.append({
        "archivo": frag["archivo"],
        "fragmento": frag["texto"]
    })

with open(NOMBRE_METADATOS, "w", encoding="utf-8") as f:
    json.dump(metadatos, f, indent=2, ensure_ascii=False)
print(f"✅ Metadatos guardados en: {NOMBRE_METADATOS}")
