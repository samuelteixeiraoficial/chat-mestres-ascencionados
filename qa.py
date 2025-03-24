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

# Link público do Google Sheets
google_sheets_csv_url = "https://docs.google.com/spreadsheets/d/1E0xHCuPXFx6TR8CgiVZvD37KizSsljT9D7eTd8lA9Aw/export?format=csv"

# Carrega e processa o CSV
@st.cache_resource
def carregar_dados():
    try:
        response = requests.get(google_sheets_csv_url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))

        perguntas_docs = []
        respostas_docs = []
        for _, row in df.iterrows():
            if pd.notna(row["Pergunta"]) and pd.notna(row["Resposta"]):
                perguntas_docs.append(Document(
                    page_content=row["Pergunta"],
                    metadata={"resposta": row["Resposta"]}
                ))
                respostas_docs.append(Document(
                    page_content=row["Resposta"]
                ))
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        db_perguntas = FAISS.from_documents(perguntas_docs, embeddings)
        db_respostas = FAISS.from_documents(respostas_docs, embeddings)
        
        return db_perguntas, db_respostas
        
    except Exception as e:
        st.error(f"Erro ao carregar o CSV: {e}")
        st.stop()

db_perguntas, db_respostas = carregar_dados()

# Carrega o template do arquivo
def carregar_template():
    try:
        with open("prompt_template.txt", "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        st.error(f"Erro ao carregar o template: {e}")
        st.stop()

template = carregar_template()

# Função para processar perguntas
def processar_pergunta(pergunta):
    try:
        # Busca contexto
        similar_perguntas = db_perguntas.similarity_search_with_score(pergunta, k=4)
        usar_respostas = all(score < 0.2 for _, score in similar_perguntas)
        
        if usar_respostas:
            contextos = [doc.page_content for doc in db_respostas.similarity_search(pergunta, k=7)]
        else:
            contextos = [doc.metadata["resposta"] for doc, _ in similar_perguntas]
        
        # Monta o prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["contexto", "pergunta"]
        ).format(
            contexto="\n".join(contextos),
            pergunta=pergunta
        )
        
        # Chama a API
        headers = {"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"}
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.5,
            "language": "pt-BR"  # Português do Brasil especificado
        }
        
        resposta = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=data
        ).json()
        
        return resposta["choices"][0]["message"]["content"]
    
    except Exception as e:
        st.error(f"Erro no processamento: {str(e)}")
        return None

# Interface principal
st.title("Chat com a Sabedoria dos Mestres Ascencionados")

# Inicializa o estado da sessão
if 'respostas' not in st.session_state:
    st.session_state.respostas = []
if 'pergunta_atual' not in st.session_state:
    st.session_state.pergunta_atual = ""
if 'processando' not in st.session_state:
    st.session_state.processando = False

# Exibe histórico de respostas
for resposta in st.session_state.respostas:
    st.markdown(
        f"<div style='margin: 20px 0; padding: 15px; border-radius: 10px; background-color: #f0f2f6;'>"
        f"<b>Resposta:</b><br>{resposta}"
        f"</div>", 
        unsafe_allow_html=True
    )

# Campo de entrada de pergunta
with st.form(key='pergunta_form'):
    col1, col2 = st.columns([5, 1])
    with col1:
        pergunta = st.text_input(
            "Sua pergunta:",
            placeholder="Escreva sua dúvida aqui...",
            key="input_pergunta",
            value=st.session_state.pergunta_atual
        )
    with col2:
        enviar = st.form_submit_button(" ⬆️ ")
    
    if enviar and pergunta.strip():
        st.session_state.pergunta_atual = pergunta
        st.session_state.processando = True

# Processamento da pergunta
if st.session_state.processando:
    with st.spinner("Processando sua pergunta..."):
        resposta = processar_pergunta(st.session_state.pergunta_atual)
        if resposta:
            st.session_state.respostas.append(resposta)
        st.session_state.processando = False
        st.session_state.pergunta_atual = ""
        st.experimental_rerun()