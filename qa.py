import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
import pandas as pd
from io import StringIO
from langchain.docstore.document import Document
import os
import time

# Carrega as variáveis de ambiente
load_dotenv()

# Configuração CSS para o botão
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #3D348B;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2A255E;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Link público do Google Sheets (exportado como CSV)
google_sheets_csv_url = "https://docs.google.com/spreadsheets/d/1E0xHCuPXFx6TR8CgiVZvD37KizSsljT9D7eTd8lA9Aw/export?format=csv"

# Carrega o CSV diretamente do link
try:
    response = requests.get(google_sheets_csv_url)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))

    perguntas_docs = []
    respostas_docs = []
    for _, row in df.iterrows():
        if pd.notna(row["Pergunta"]) and pd.notna(row["Resposta"]):
            perguntas_docs.append(Document(page_content=row["Pergunta"], metadata={"resposta": row["Resposta"]}))
            respostas_docs.append(Document(page_content=row["Resposta"]))
except Exception as e:
    st.error(f"Erro ao carregar o CSV: {e}")

# Configuração dos embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_perguntas = FAISS.from_documents(perguntas_docs, embeddings)
db_respostas = FAISS.from_documents(respostas_docs, embeddings)

# Funções de busca e API
def retrieve_info(query):
    similar_perguntas = db_perguntas.similarity_search_with_score(query, k=4)
    usar_respostas = all(score < 0.2 for _, score in similar_perguntas)
    return [doc.metadata["resposta"] for doc, _ in similar_perguntas] if not usar_respostas else [doc.page_content for doc in db_respostas.similarity_search(query, k=7)]

def call_deepseek_api(prompt):
    try:
        headers = {"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"}
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.5,
            "language": "pt"
        }
        return requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=data).json()
    except Exception as e:
        st.error(f"Erro na API: {e}")
        return None

# Animação de carregamento
def animacao_carregamento():
    placeholder = st.empty()
    dots = ""
    while st.session_state.processando:
        dots = dots + "." if len(dots) < 3 else ""
        placeholder.markdown(f"**Processando**{dots}")
        time.sleep(0.5)
    placeholder.empty()

# Interface do Streamlit
st.title("Chat com a Sabedoria dos Mestres Ascencionados")

# Gerenciamento de estado
if "respostas" not in st.session_state:
    st.session_state.respostas = []
if "processando" not in st.session_state:
    st.session_state.processando = False

# Exibe respostas anteriores
for resposta in st.session_state.respostas:
    st.markdown(f"<div style='margin: 20px 0; padding: 15px; border-radius: 10px; background-color: #f0f2f6;'><b>Resposta:</b><br>{resposta}</div>", unsafe_allow_html=True)

# Campo de pergunta (posição dinâmica)
def campo_pergunta():
    col1, col2 = st.columns([5, 1])
    with col1:
        pergunta = st.text_input("Sua pergunta:", key="input_pergunta", placeholder="Escreva sua dúvida espiritual aqui...")
    with col2:
        if st.button("🌀 Enviar", key="btn_enviar"):
            if pergunta.strip():
                st.session_state.processando = True
                st.session_state.respostas.append("")  # Placeholder para resposta
                st.session_state.pergunta_atual = pergunta
            else:
                st.error("Por favor, insira uma pergunta válida.")
    return pergunta

# Lógica de processamento
if st.session_state.get("processando", False):
    animacao_carregamento()
    
    # Processa a pergunta
    contextos = retrieve_info(st.session_state.pergunta_atual)
    prompt = PromptTemplate(
        template=open("prompt_template.txt").read(),  # Arquivo com seu template
        input_variables=["contexto", "pergunta"]
    ).format(contexto="\n".join(contextos), pergunta=st.session_state.pergunta_atual)
    
    resposta = call_deepseek_api(prompt)
    if resposta:
        st.session_state.respostas[-1] = resposta["choices"][0]["message"]["content"]
    
    st.session_state.processando = False
    st.experimental_rerun()

# Posiciona o campo de pergunta
if len(st.session_state.respostas) > 0:
    campo_pergunta()
else:
    campo_pergunta()