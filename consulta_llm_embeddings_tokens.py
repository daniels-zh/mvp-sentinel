import json
import faiss
import numpy as np
import os
import csv
import ast
import time
from openai import OpenAI
from transformers import GPT2TokenizerFast
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAIError
from tqdm import tqdm
from datetime import datetime


# Cargar variables de entorno
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tokenizador para estimar tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def estimar_tokens(texto):
    return len(tokenizer.encode(texto))

# Cargar metadatos y FAISS
with open("metadatos_fragments.json", "r", encoding="utf-8") as f:
    metadatos = json.load(f)

fragments = [item["fragmento"] for item in metadatos]
embeddings = np.array([item["embedding"] for item in metadatos], dtype=np.float32)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def generar_embedding_openai(texto):
    response = client.embeddings.create(
        input=texto,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding



def guardar_metrica(data):
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    archivo = os.path.join(ruta_script, "metricas_uso.csv")

    try:
        existe = os.path.isfile(archivo)
        print(f"📝 Guardando métrica en: {archivo}")
        with open(archivo, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if not existe:
                print("📄 El archivo no existe, se creará con encabezados.")
                writer.writeheader()
            writer.writerow(data)
        print("✅ Métrica guardada correctamente.")
    except Exception as e:
        print(f"❌ Error al guardar la métrica: {e}")

def consultar_llm(consulta, k=3):
    emb_consulta = generar_embedding_openai(consulta)
    _, indices = index.search(np.array([emb_consulta]).astype("float32"), k)

    fragmentos_seleccionados = [fragments[idx] for idx in indices[0]]
    tokens_estimados = sum(estimar_tokens(frag) for frag in fragmentos_seleccionados)

    print("\n📌 Fragmentos seleccionados:")
    for i, frag in enumerate(fragmentos_seleccionados):
        print(f"\n🔹 Fragmento {i+1} ({estimar_tokens(frag)} tokens):\n{frag[:300]}...")

    contexto = "\n\n".join(fragmentos_seleccionados)
    prompt = f"Contexto:\n{contexto}\n\nResponde a: {consulta}"

    print(f"\n🔢 Tokens estimados del prompt: {estimar_tokens(prompt)}")

    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres un asistente experto en seguridad operacional en minería subterránea."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=600
    )
    end_time = time.time()
    tiempo_respuesta = round(end_time - start_time, 2)  # tiempo en segundos

    usage = response.usage
    print(f"\n📊 Tokens usados: prompt={usage.prompt_tokens}, respuesta={usage.completion_tokens}, total={usage.total_tokens}")

    # 💾 Guardar métricas antes de retornar
    guardar_metrica({
        "fecha_hora": datetime.now().isoformat(),
        "consulta": consulta,
        "tokens_prompt": usage.prompt_tokens,
        "tokens_respuesta": usage.completion_tokens,
        "tokens_totales": usage.total_tokens,
        "tiempo_respuesta_seg": tiempo_respuesta 
    })
    return response.choices[0].message.content

def similitud_coseno(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def buscar_en_cache_semantica(pregunta, umbral_similitud=0.85):
    archivo = "cache_respuestas.csv"
    if not os.path.isfile(archivo):
        return None

    try:
        emb_consulta = generar_embedding_openai(pregunta)
    except OpenAIError:
        return None

    with open(archivo, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for fila in reader:
            emb_guardado = ast.literal_eval(fila["embedding"])
            similitud = similitud_coseno(emb_consulta, emb_guardado)
            print(f"🔍 Similitud encontrada: {similitud:.2f}")
            if similitud >= umbral_similitud:
                print(f"🔁 Respuesta recuperada del cache (similitud: {similitud:.2f})")
                return fila["respuesta"]
    return None


def guardar_en_cache(pregunta, respuesta):
    archivo = "cache_respuestas.csv"
    existe = os.path.isfile(archivo)
    embedding = generar_embedding_openai(pregunta)

    with open(archivo, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["pregunta", "respuesta", "embedding"])
        if not existe:
            writer.writeheader()
        writer.writerow({
            "pregunta": pregunta,
            "respuesta": respuesta,
            "embedding": str(embedding)
        })

if __name__ == "__main__":
    print("💬 Ingresa tu consulta (o escribe 'salir'):")
    while True:
        consulta = input("\nConsulta: ")
        if consulta.lower() == "salir":
            break
        #respuesta = consultar_llm(consulta)
        respuesta = buscar_en_cache_semantica(consulta)
        if not respuesta:
            respuesta = consultar_llm(consulta)
            guardar_en_cache(consulta, respuesta)
        print("\n🧠 Respuesta del modelo:")
        print(respuesta)


