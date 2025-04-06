import streamlit as st
from dotenv import load_dotenv
from functions import carregar_dados, carregar_template, processar_pergunta
import os

# Carrega as variáveis de ambiente
load_dotenv()

# Link do Google Sheets CSV
google_sheets_csv_url = "https://docs.google.com/spreadsheets/d/1E0xHCuPXFx6TR8CgiVZvD37KizSsljT9D7eTd8lA9Aw/export?format=csv"

# Carregar o arquivo CSS (estilo visual do app)
with open("styles.css", "r", encoding="utf-8") as file:
    st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)

# Carregar dados com cache
@st.cache_resource
def carregar_dados_cached():
    return carregar_dados(google_sheets_csv_url)

# Carregar template com cache
@st.cache_data
def carregar_template_cached():
    return carregar_template("prompt_template.txt")

# Carregar os dados da planilha
db_perguntas, db_respostas = carregar_dados_cached()

# Carregar o template de prompt
template = carregar_template_cached()

# Título do app
st.title("Chat com a Sabedoria dos Mestres Ascencionados")

# Inicializa o histórico da sessão
if 'historico' not in st.session_state:
    st.session_state.historico = []

# Exibir histórico de perguntas e respostas
for mensagem in st.session_state.historico:
    pergunta, resposta = mensagem["pergunta"], mensagem["resposta"]
    st.markdown(f"""
        <div class='mensagem-box pergunta-box' style="text-align:right;">🙃 {pergunta}</div>
        <div class='mensagem-box resposta-box' style="text-align:left;">👽 {resposta}</div>
    """, unsafe_allow_html=True)

# Formulário para nova pergunta
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
            resposta = processar_pergunta(
                pergunta,
                db_perguntas,
                db_respostas,
                template,
                os.getenv("DEEPSEEK_API_KEY")
            )
            if resposta:
                st.session_state.historico.append({"pergunta": pergunta, "resposta": resposta})
                st.rerun()

# Aviso de responsabilidade
st.markdown(
    "<p class='aviso'>Este AI-Chat pode cometer erros. Verifique informações importantes.</p>",
    unsafe_allow_html=True
)
