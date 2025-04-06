import streamlit as st
from dotenv import load_dotenv
import os
import logging
import requests
import pandas as pd
import sys
from io import StringIO

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carrega variáveis de ambiente
load_dotenv()

# URL da planilha
google_sheets_csv_url = "https://docs.google.com/spreadsheets/d/1E0xHCuPXFx6TR8CgiVZvD37KizSsljT9D7eTd8lA9Aw/export?format=csv"

# CSS
with open("styles.css", "r") as file:
    st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)

# Carregar dados com debug
@st.cache_resource(ttl=3600)
def carregar_dados_csv():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Encoding": "gzip"
        }
        logger.info("Requisição do CSV...")
        response = requests.get(google_sheets_csv_url, headers=headers, timeout=30)
        response.raise_for_status()

        logger.info("Convertendo CSV para DataFrame...")
        csv_content = response.content.decode("utf-8")
        df = pd.read_csv(StringIO(csv_content))

        if df.empty:
            raise ValueError("⚠️ DataFrame retornado está vazio!")

        # Verifica colunas obrigatórias
        colunas_esperadas = {"Pergunta", "Resposta"}
        if not colunas_esperadas.issubset(df.columns):
            raise ValueError(f"⚠️ Colunas esperadas não encontradas! Colunas atuais: {df.columns.tolist()}")

        return df

    except Exception as e:
        st.error("❌ Erro ao carregar os dados.")
        st.exception(e)
        logger.error(f"Erro no carregamento do CSV: {e}", exc_info=True)
        st.stop()

# Load CSV
try:
    df = carregar_dados_csv()
    st.success("✅ Base carregada com sucesso!")
except Exception as e:
    st.error("❌ Erro ao carregar dados.")
    st.stop()

# Debug info expandido
with st.expander("🔍 Debug Info", expanded=True):
    st.subheader("Pré-visualização da base")
    st.write(df.head())
    st.write("Colunas:", df.columns.tolist())
    st.write("Total de linhas:", len(df))

    st.subheader("Variáveis de Ambiente")
    st.write(dict(os.environ))

    st.subheader("Informações do Sistema")
    st.write("Sistema operacional:", os.name)
    st.write("Versão Python:", sys.version)
    st.write("Diretório atual:", os.getcwd())
    st.write("Arquivos no diretório:", os.listdir())

# Interface principal
st.title("Chat com a Sabedoria dos Mestres Ascencionados")

# Simples exibição das perguntas
st.markdown("### 🧠 Perguntas carregadas da base:")
for i, row in df.iterrows():
    st.markdown(f"**Q:** {row['Pergunta']}\n\n**A:** {row['Resposta']}")
    if i >= 3:
        break
