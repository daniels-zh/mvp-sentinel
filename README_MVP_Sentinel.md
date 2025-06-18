# 🛠️ MVP Sentinel - Consulta Inteligente de Documentos

Este repositorio contiene el prototipo funcional del **MVP Sentinel**, una herramienta de consulta inteligente basada en lenguaje natural que permite a los usuarios realizar preguntas sobre documentación interna mediante el uso de embeddings, FAISS y modelos de lenguaje como GPT-4.

---

## 🚀 Descripción del Proyecto

MVP Sentinel permite realizar consultas en lenguaje natural sobre documentos PDF institucionales previamente cargados, procesados y vectorizados. Utiliza un enfoque de **RAG (Retrieval-Augmented Generation)** para entregar respuestas relevantes con base en fragmentos documentales.

---

## ⚙️ Arquitectura Técnica

- **Procesamiento inicial:** Los PDF se transforman a texto (.txt) y se segmentan en fragmentos.
- **Embeddings:** Se generan vectores usando el modelo `paraphrase-MiniLM-L6-v2`.
- **FAISS:** Se construye un índice vectorial para realizar búsquedas semánticas rápidas.
- **OpenAI GPT-4:** Se usa para generar respuestas basadas en los fragmentos recuperados.
- **Interfaz web:** Implementada con `Streamlit` y desplegada en la nube.

---

## 📁 Estructura del repositorio

```
MVP_Sentinel_WEB/
│
├── app.py                          # Interfaz Streamlit
├── generate_embeddings_filtrado.py# Script para generar FAISS con MiniLM
├── process_pdfs.py                # Procesamiento de PDF a TXT
├── consulta_llm_embeddings_tokens.py # Motor de consulta local
├── vector_index_minilm.faiss      # Índice FAISS generado
├── metadatos_minilm.json          # Metadatos por fragmento
├── docs/                          # Archivos PDF originales (no incluidos en Git)
├── context/                       # Archivos TXT extraídos
├── requirements.txt               # Dependencias del proyecto
└── cache_respuestas.csv           # (Opcional) historial de respuestas
```

---

## 🧪 Cómo ejecutar localmente

1. Clona el repositorio:
```bash
git clone https://github.com/daniels-zh/mvp-sentinel.git
cd mvp-sentinel
```

2. Crea y activa un entorno virtual:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Instala dependencias:
```bash
pip install -r requirements.txt
```

4. Ejecuta la app:
```bash
streamlit run app.py
```

> Asegúrate de tener las variables de entorno definidas en `.env` o usar `secrets.toml` en Streamlit Cloud para `OPENAI_API_KEY`.

---

## 🌐 App en producción

Puedes probar la aplicación desplegada en:
🔗 [https://mvp-sentinel-gmpox6pyvyjn8bze64b8vh.streamlit.app](https://mvp-sentinel-gmpox6pyvyjn8bze64b8vh.streamlit.app)

---

## 📋 Ejemplos de uso

- `¿Qué temas abordan los boletines de seguridad del 2025?`
- `¿Qué incidentes relevantes se reportaron en marzo de 2025?`
- `¿Cuáles fueron las medidas correctivas en el Proyecto ESSL?`

---

## 📄 Licencia

Este proyecto es parte de un ejercicio académico de innovación tecnológica y no está destinado aún para uso comercial.

---

## 🙋‍♂️ Contacto

Desarrollado por Daniel Zamorano Hernández  
Repositorio GitHub: [daniels-zh](https://github.com/daniels-zh)
