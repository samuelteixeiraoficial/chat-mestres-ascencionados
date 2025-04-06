import streamlit as st
from dotenv import load_dotenv
from functions import carregar_dados, carregar_template, processar_pergunta, verificar_dados
import os
import logging
import requests
import pandas as pd
import sys
from io import StringIO

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carrega variáveis de ambiente (.env)
load_dotenv()

# Link do Google Sheets (CSV exportado publicamente)
google_sheets_csv_url = "https://docs.google.com/spreadsheets/d/1E0xHCuPXFx6TR8CgiVZvD37KizSsljT9D7eTd8lA9Aw/export?format=csv"

# Carregar CSS
try:
    with open("styles.css", "r") as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)
except Exception as e:
    st.warning("⚠️ Não foi possível carregar o CSS.")

# Teste de conexão com Google Sheets (antes de tudo!)
with st.expander("🔍 Debug Info", expanded=True):
    st.subheader("Teste de conexão com Google Sheets")
    try:
        response = requests.get(google_sheets_csv_url, timeout=10)
        response.raise_for_status()
        st.success(f"✅ Conexão bem-sucedida! Status: {response.status_code}")
        st.code(response.text[:500])  # Mostra os primeiros caracteres
    except Exception as e:
        st.error("❌ Falha ao conectar ao Google Sheets.")
        st.exception(e)

    st.subheader("Variáveis de Ambiente")
    st.write(dict(os.environ))

    st.subheader("Informações do Sistema")
    st.write("Sistema operacional:", os.name)
    st.write("Versão Python:", sys.version)
    st.write("Diretório atual:", os.getcwd())
    st.write("Arquivos no diretório:", os.listdir())

# Cache de carregamento de dados
@st.cache_resource(ttl=3600)
def carregar_dados_cached():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Encoding": "gzip"
        }
        logger.info("Iniciando download do CSV...")
        response = requests.get(google_sheets_csv_url, headers=headers, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text), encoding='utf-8')
        if df.empty:
            logger.error("DataFrame vazio!")
            raise ValueError("DataFrame vazio")

        return carregar_dados(google_sheets_csv_url)
    except Exception as e:
        st.error("❌ Erro ao carregar os dados da planilha.")
        st.exception(e)
        logger.error(f"ERRO no carregamento: {str(e)}", exc_info=True)
        st.stop()

# Cache do template
@st.cache_data
def carregar_template_cached():
    return carregar_template("prompt_template.txt")

# Iniciar carregamento dos dados
try:
    logger.info("Iniciando carregamento de dados...")
    with st.spinner('📚 Carregando base de conhecimento...'):
        db_perguntas, db_respostas = carregar_dados_cached()
    verificar_dados(db_perguntas, db_respostas)
    logger.info("Dados carregados e verificados com sucesso!")
except Exception as e:
    logger.error(f"Erro ao carregar dados: {str(e)}", exc_info=True)
    st.error("Erro ao carregar os dados. Recarregue a página ou tente mais tarde.")
    st.stop()

# Carregar o template
try:
    template = carregar_template_cached()
except Exception as e:
    logger.error(f"Erro ao carregar template: {str(e)}", exc_info=True)
    st.error("Erro ao carregar o template do sistema.")
    st.stop()

# Interface principal
st.title("Chat com a Sabedoria dos Mestres Ascencionados")

if 'historico' not in st.session_state:
    st.session_state.historico = []

# Histórico
for mensagem in st.session_state.historico:
    pergunta, resposta = mensagem["pergunta"], mensagem["resposta"]
    st.markdown(f"""
        <div class='mensagem-box pergunta-box' style="text-align:right;">🙃 {pergunta}</div>
        <div class='mensagem-box resposta-box' style="text-align:left;">👽 {resposta}</div>
    """, unsafe_allow_html=True)

# Formulário de pergunta
with st.form(key='pergunta_form'):
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1])
    with col1:
        pergunta = st.text_input("Sua pergunta:", placeholder="Escreva sua dúvida aqui...", key="input_pergunta")
    with col2:
        enviar = st.form_submit_button(" ⬆️ ")
    st.markdown('</div>', unsafe_allow_html=True)

# Processar a pergunta
if enviar and pergunta.strip():
    with st.spinner("Digitando..."):
        resposta = processar_pergunta(pergunta, db_perguntas, db_respostas, template, os.getenv("DEEPSEEK_API_KEY"))
        if resposta:
            st.session_state.historico.append({"pergunta": pergunta, "resposta": resposta})
            st.rerun()

# Rodapé
st.markdown("<p class='aviso'>Este AI-Chat pode cometer erros. Verifique informações importantes.</p>", unsafe_allow_html=True)
