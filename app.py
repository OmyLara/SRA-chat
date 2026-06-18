import os
import json
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Configuración de la página web
st.set_page_config(page_title="SwimRight Assistant", page_icon="🏊🏼‍♂️", layout="centered")

def cargar_variables_desde_env():
    """Bypass manual para leer el archivo .env"""
    ruta_env = ".env"
    if os.path.exists(ruta_env):
        with open(ruta_env, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if linea and not linea.startswith("#") and "=" in linea:
                    clave, valor = linea.split("=", 1)
                    os.environ[clave.strip()] = valor.strip()

# Cargar llaves al iniciar la app
cargar_variables_desde_env()

# Inicializar componentes de IA en la caché de Streamlit para que no se recarguen en cada clic
@st.cache_resource
def inicializar_rag():
    persist_directory = "./chroma_db"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    return vector_db, llm

try:
    vector_db, llm = inicializar_rag()
except Exception as e:
    st.error(f"Error al inicializar las APIs. Revisa tu archivo .env: {e}")
    st.stop()

# --- INTERFAZ DE USUARIO (UX) ---
st.markdown("""<h1 style="color: #0166CC; font-size: 44px; text-align: center">Ask me anything about your swimming lessons!</h1>""",
            unsafe_allow_html=True)
#st.subheader("¿Qué quieres saber hoy, mi ludópata querido?")

# Inicializar el historial de chat en la sesión si no existe
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! Ready to make a splash? Ask me anything about swim lessons, class schedules, levels, or pool information."}
    ]

# Mostrar los mensajes históricos en la pantalla con burbujas de diseño
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capturar la entrada del usuario en la caja de texto estilo ChatGPT
if pregunta_usuario := st.chat_input("What days and times are the swim lessons?"):
    
    # 1. Mostrar inmediatamente el mensaje del usuario en la pantalla
    with st.chat_message("user"):
        st.markdown(pregunta_usuario)
    st.session_state.messages.append({"role": "user", "content": pregunta_usuario})

    # 2. Buscar en ChromaDB el fragmento más relevante
    docs = vector_db.similarity_search(pregunta_usuario, k=1)
    contexto = docs[0].page_content if docs else "No hay contexto disponible."

    # 3. Construir el prompt estructurado con tu personalidad
    prompt_estructurado = f"""
    You are a helpful assistant for a swim school. Answer the user's question using ONLY the context provided below.
    You gotta be profesional and friendly since we have a lot of kids and parents asking questions. If the context uses a friendly or casual tone (e.g., Hi there! or Hey friend! or similar ones. but after the first interaction don't repeat the same starting phrase), you should match that tone when responding. For SwimRight Academy, always keep responses warm, supportive, and parent-friendly, especially when talking about swim lessons, children progress, or water safety.
    Always ask for the location to show the right information.
    {contexto}

    User's question:
    {pregunta_usuario}
    """

    # 4. Generar la respuesta usando Gemini con un efecto visual de carga (Spinner)
    with st.chat_message("assistant"):
        with st.spinner("Diving into the details... 🌊"):
            try:
                respuesta_gemini = llm.invoke(prompt_estructurado)
                respuesta_texto = respuesta_gemini.content
                st.markdown(respuesta_texto)
                
                # Guardar la respuesta en el historial
                st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})
            except Exception as e:
                st.error(f"Hubo un error con la API: {e}")