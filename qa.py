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

# Carrega o CSS externo
with open("styles.css", "r") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

# Link do Google Sheets
google_sheets_csv_url = "https://docs.google.com/spreadsheets/d/1E0xHCuPXFx6TR8CgiVZvD37KizSsljT9D7eTd8lA9Aw/export?format=csv"


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


def carregar_template():
    try:
        with open("prompt_template.txt", "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        st.error(f"Erro ao carregar o template: {e}")
        st.stop()


template = carregar_template()


def processar_pergunta(pergunta):
    try:
        similar_perguntas = db_perguntas.similarity_search_with_score(
            pergunta, k=4)
        usar_respostas = all(score < 0.2 for _, score in similar_perguntas)

        if usar_respostas:
            contextos = [
                doc.page_content for doc in db_respostas.similarity_search(pergunta, k=7)]
        else:
            contextos = [doc.metadata["resposta"]
                         for doc, _ in similar_perguntas]

        prompt = PromptTemplate(
            template=template,
            input_variables=["contexto", "pergunta"]
        ).format(
            contexto="\n".join(contextos),
            pergunta=pergunta
        )

        headers = {"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"}
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.5,
            "language": "pt-BR"
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

if 'historico' not in st.session_state:
    st.session_state.historico = []

# Exibir histórico de perguntas e respostas no formato de chat
for mensagem in st.session_state.historico:
    pergunta, resposta = mensagem["pergunta"], mensagem["resposta"]
    st.markdown(f"""
        <div class='mensagem-box pergunta-box'>👤 {pergunta}</div>
        <div class='mensagem-box resposta-box'>🤖 {resposta}</div>
    """, unsafe_allow_html=True)

# Formulário de entrada
with st.form(key='pergunta_form'):
    st.markdown("<div class='form-container'>", unsafe_allow_html=True)

    col1, col2 = st.columns([5, 1]) if st.session_state.get(
        "tela_grande", True) else (st.container(), st.container())

    with col1:
        pergunta = st.text_input(
            "Sua pergunta:",
            placeholder="Escreva sua dúvida espiritual aqui...",
            key="input_pergunta"
        )

    with col2:
        enviar = st.form_submit_button("🌀 Enviar")

    st.markdown("</div>", unsafe_allow_html=True)

# Adiciona o aviso abaixo do campo de pergunta
st.markdown("<p class='aviso'>Este AI-Chat pode cometer erros. Verifique informações importantes.</p>",
            unsafe_allow_html=True)

# Processamento da pergunta
if enviar and pergunta.strip():
    st.session_state.pergunta_atual = pergunta
    st.session_state.processando = True

    # Reseta o campo de entrada
    st.session_state.input_pergunta = ""  # Limpa o texto da pergunta anterior
    st.rerun()  # Atualiza a interface imediatamente
