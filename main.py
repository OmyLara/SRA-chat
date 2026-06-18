import json
import os
from langchain_openai import OpenAIEmbeddings
###IN CASE I USE GEMINI###
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def cargar_variables_desde_env():
    """Bypass manual para leer el archivo .env sin usar la librería python-dotenv"""
    ruta_env = ".env"
    if os.path.exists(ruta_env):
        with open(ruta_env, "r", encoding="utf-8") as f:
            for linea in f:
                # Limpiar la línea y omitir comentarios o líneas vacías
                linea = linea.strip()
                if linea and not linea.startswith("#") and "=" in linea:
                    clave, valor = linea.split("=", 1)
                    # Guardar la llave en el entorno del sistema de Python
                    os.environ[clave.strip()] = valor.strip()
        print("🔑 API Keys cargadas manualmente desde el archivo .env")
    else:
        print("⚠️ Advertencia: No se encontró el archivo .env")

def cargar_datos_a_vector_db():
    # 1. Cargar las llaves usando nuestro bypass manual
    cargar_variables_desde_env()
    
    ruta_jsonl = "data.jsonl"
    if not os.path.exists(ruta_jsonl):
        print(f"❌ Error: No se encontró el archivo '{ruta_jsonl}' en esta carpeta.")
        return

    print("📖 Leyendo tu archivo JSONL de juegos de mesa...")
    documentos = []
    
    # 2. Leer el archivo JSONL
    with open(ruta_jsonl, "r", encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                data = json.loads(linea)
                texto_completo = f"Pregunta: {data['prompt']}\nRespuesta: {data['response']}"
                
                doc = Document(
                    page_content=texto_completo,
                    metadata={"fuente": "juegos_jsonl"}
                )
                documentos.append(doc)

    print(f"✅ ¡Éxito! Se procesaron {len(documentos)} interacciones.")

    # 3. Inicializar el creador de vectores (Embeddings) de OpenAI
    print("🧠 Convirtiendo tus textos en vectores matemáticos (LLM Embeddings)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    

    # 4. Crear la base de datos Chroma localmente
    persist_directory = "./chroma_db"
    print(f"💾 Guardando los vectores en la carpeta local '{persist_directory}'...")
    
    vector_db = Chroma.from_documents(
        documents=documentos,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print("\n🎉 ¡PROCESO TERMINADO CON ÉXITO!")
    print(f"Se ha creado la carpeta '{persist_directory}'.")

if __name__ == "__main__":
    cargar_datos_a_vector_db()