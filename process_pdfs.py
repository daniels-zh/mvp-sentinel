import os
import re
from PyPDF2 import PdfReader
from pathlib import Path

RUTA_PDFS = "docs"
RUTA_SALIDA = "context"

def extraer_texto(pdf_path):
    texto = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            texto += page.extract_text() + "\n"
    except Exception as e:
        print(f"⚠️ Error leyendo {pdf_path}: {e}")
    return texto

def limpiar_y_segmentar(texto, longitud_max=500):
    # Limpia saltos y espacios repetidos
    texto = re.sub(r'\n+', '\n', texto)
    texto = re.sub(r'[ \t]+', ' ', texto)
    texto = texto.strip()

    # Divide el texto en oraciones usando puntuación
    oraciones = re.split(r'(?<=[\.\:\!\?])\s+', texto)
    
    fragmentos = []
    buffer = ""

    for oracion in oraciones:
        if len(buffer) + len(oracion) < longitud_max:
            buffer += oracion + " "
        else:
            fragmentos.append(buffer.strip())
            buffer = oracion + " "
    if buffer:
        fragmentos.append(buffer.strip())

    return fragmentos

def guardar_fragmentos(nombre_archivo, fragmentos):
    base = os.path.splitext(nombre_archivo)[0]
    ruta_salida = os.path.join(RUTA_SALIDA, f"{base}.txt")
    with open(ruta_salida, "w", encoding="utf-8") as f:
        for fragmento in fragmentos:
            f.write(fragmento + "\n\n")

if __name__ == "__main__":
    os.makedirs(RUTA_SALIDA, exist_ok=True)
    archivos_pdf = [f for f in os.listdir(RUTA_PDFS) if f.lower().endswith(".pdf")]

    for archivo in archivos_pdf:
        ruta_pdf = os.path.join(RUTA_PDFS, archivo)
        texto = extraer_texto(ruta_pdf)
        if texto:
            fragmentos = limpiar_y_segmentar(texto)
            guardar_fragmentos(archivo, fragmentos)
            print(f"✅ Convertido y segmentado: {archivo} -> {len(fragmentos)} fragmentos")
        else:
            print(f"⚠️ No se extrajo texto de: {archivo}")