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

# Carrega as variáveis de ambiente
load_dotenv()

# Link do Google Sheets
google_sheets_csv_url = "https://docs.google.com/spreadsheets/d/1E0xHCuPXFx6TR8CgiVZvD37KizSsljT9D7eTd8lA9Aw/export?format=csv"

# Carregar o arquivo CSS
with open("styles.css", "r") as file:
    st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)

# Carregar dados com cache (com timeout aumentado e headers)
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
        
        df = pd.read_csv(StringIO(response.text))
        if df.empty:
            logger.error("DataFrame vazio!")
            return None, None
            
        return carregar_dados(google_sheets_csv_url)
    except Exception as e:
        logger.error(f"ERRO no carregamento: {str(e)}", exc_info=True)
        raise

# Carregar template com cache
@st.cache_data
def carregar_template_cached():
    return carregar_template("prompt_template.txt")

# Carregar e verificar os dados COM TRATAMENTO DE ERRO
try:
    logger.info("Iniciando carregamento de dados...")
    with st.spinner('Carregando base de conhecimento... (pode demorar alguns segundos)'):
        db_perguntas, db_respostas = carregar_dados_cached()
    
    verificar_dados(db_perguntas, db_respostas)
    logger.info("Dados verificados com sucesso!")
    
except Exception as e:
    logger.error(f"Erro ao carregar dados: {str(e)}", exc_info=True)
    st.error("""
        Erro ao carregar os dados. 
        Por favor, recarregue a página. 
        Se o problema persistir, tente novamente mais tarde.
    """)
    st.stop()

# Carregar o template
try:
    template = carregar_template_cached()
except Exception as e:
    logger.error(f"Erro ao carregar template: {str(e)}")
    st.error("Erro ao carregar o template do sistema.")
    st.stop()

# DEBUG: Verificação completa (remova depois)
with st.expander("🔍 Debug Info", expanded=False):
    st.subheader("Teste de Requisição")
    try:
        test_req = requests.get(google_sheets_csv_url, timeout=10)
        st.write(f"Status: {test_req.status_code}")
        st.write(f"Tamanho: {len(test_req.text)} bytes")
        st.code(f"Primeiras linhas:\n{test_req.text[:200]}...")
    except Exception as e:
        st.error(f"Falha na requisição: {str(e)}")
    
    st.subheader("Variáveis de Ambiente")
    st.write(dict(os.environ))
    
    st.subheader("Informações do Sistema")
    st.write("Sistema operacional:", os.name)
    st.write("Versão Python:", sys.version)
    st.write("Diretório atual:", os.getcwd())
    st.write("Arquivos no diretório:", os.listdir())

# Interface principal
st.title("Chat com a Sabedoria dos Mestres Ascencionados")

if 'historico' not in st.session_state:
    st.session_state.historico = []

# Exibir histórico de perguntas e respostas
for mensagem in st.session_state.historico:
    pergunta, resposta = mensagem["pergunta"], mensagem["resposta"]
    st.markdown(f"""
        <div class='mensagem-box pergunta-box' style="text-align:right;">🙃 {pergunta}</div>
        <div class='mensagem-box resposta-box' style="text-align:left;">👽 {resposta}</div>
    """, unsafe_allow_html=True)

# Formulário de entrada
with st.form(key='pergunta_form'):
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1])
    with col1:
        pergunta = st.text_input(
            "Sua pergunta:",
            placeholder="Escreva sua dúvida aqui...",
            key="input_pergunta"
        )
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
st.markdown("<p class='aviso'>Este AI-Chat pode cometer erros. Verifique informações importantes.</p>",
            unsafe_allow_html=True)