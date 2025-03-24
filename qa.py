import streamlit as st
from dotenv import load_dotenv
from functions import carregar_dados, carregar_template, processar_pergunta
import os

# Carrega as variáveis de ambiente
load_dotenv()

# Link do Google Sheets
google_sheets_csv_url = "https://docs.google.com/spreadsheets/d/1E0xHCuPXFx6TR8CgiVZvD37KizSsljT9D7eTd8lA9Aw/export?format=csv"

# Carregar o arquivo CSS
with open("styles.css", "r") as file:
    st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)


# Carregar dados com cache
@st.cache_resource
def carregar_dados_cached():
    return carregar_dados(google_sheets_csv_url)

# Carregar template com cache
@st.cache_data
def carregar_template_cached():
    return carregar_template("prompt_template.txt")

# Carregar os dados
db_perguntas, db_respostas = carregar_dados_cached()

# Carregar o template
template = carregar_template_cached()

# Interface principal
st.title("Chat com a Sabedoria dos Mestres Ascencionados")

if 'historico' not in st.session_state:
    st.session_state.historico = []

# Exibir histórico de perguntas e respostas no formato de chat
for mensagem in st.session_state.historico:
    pergunta, resposta = mensagem["pergunta"], mensagem["resposta"]
    st.markdown(f"""
        <div class='mensagem-box pergunta-box' style="text-align:right;">🙃 {pergunta}</div>
        <div class='mensagem-box resposta-box' style="text-align:left;">👽 {resposta}</div>
    """, unsafe_allow_html=True)

# Formulário de entrada
with st.form(key='pergunta_form'):
    col1, col2 = st.columns([5, 1])
    
    with col1:
        pergunta = st.text_input(
            "Sua pergunta:",
            placeholder="Escreva sua dúvida aqui...",
            key="input_pergunta"
        )
    
    with col2:
        st.markdown("<div style='display: flex; align-items: center; height: 100%;'>", unsafe_allow_html=True)
        enviar = st.form_submit_button(" ⬆️ ")
        st.markdown("</div>", unsafe_allow_html=True)

    if enviar and pergunta.strip():
        with st.spinner("Digitando..."):
            resposta = processar_pergunta(pergunta, db_perguntas, db_respostas, template, os.getenv("DEEPSEEK_API_KEY"))
            if resposta:
                st.session_state.historico.append({"pergunta": pergunta, "resposta": resposta})
                st.rerun()  # Rerun mais eficiente

# Adiciona o aviso abaixo do campo de pergunta
st.markdown("<p class='aviso'>Este AI-Chat pode cometer erros. Verifique informações importantes.</p>",
            unsafe_allow_html=True)