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

# Carrega as variáveis de ambiente
load_dotenv()

# Link público do Google Sheets (exportado como CSV)
google_sheets_csv_url = "https://docs.google.com/spreadsheets/d/1E0xHCuPXFx6TR8CgiVZvD37KizSsljT9D7eTd8lA9Aw/export?format=csv"

# Carrega o CSV diretamente do link
try:
    # Faz o download do CSV
    response = requests.get(google_sheets_csv_url)
    response.raise_for_status()

    # Usa o pandas para ler o CSV
    df = pd.read_csv(StringIO(response.text))

    # Converte as colunas "Pergunta" e "Resposta" para documentos
    perguntas_docs = []
    respostas_docs = []
    for _, row in df.iterrows():
        # Verifica se a pergunta e a resposta não são NaN
        if pd.notna(row["Pergunta"]) and pd.notna(row["Resposta"]):
            perguntas_docs.append(
                Document(page_content=row["Pergunta"], metadata={"resposta": row["Resposta"]})
            )
            respostas_docs.append(Document(page_content=row["Resposta"]))

except Exception as e:
    st.error(f"Erro ao carregar o CSV: {e}")

# Configuração dos embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Cria dois bancos de dados FAISS
db_perguntas = FAISS.from_documents(perguntas_docs, embeddings)  # Para buscar perguntas
db_respostas = FAISS.from_documents(respostas_docs, embeddings)  # Para buscar respostas

# Função para buscar documentos semelhantes
def retrieve_info(query):
    # Busca as 4 perguntas mais semelhantes com suas pontuações de similaridade
    similar_perguntas = db_perguntas.similarity_search_with_score(query, k=4)
    
    # Verifica se alguma pergunta tem similaridade >= 20% (ajuste o limiar conforme necessário)
    usar_respostas = True
    for doc, score in similar_perguntas:
        if score >= 0.2:  # Se a similaridade for maior ou igual a 20%
            usar_respostas = False
            break
    
    if not usar_respostas:
        # Método 2: Usa as respostas das perguntas semelhantes
        contextos = [doc.metadata["resposta"] for doc, _ in similar_perguntas]
    else:
        # Método 1: Busca diretamente nas respostas
        similar_respostas = db_respostas.similarity_search(query, k=7)
        contextos = [doc.page_content for doc in similar_respostas]
    
    return contextos

# Configuração da API da DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# Template para o prompt
template = """
Você é um assistente virtual de uma egrégora de seres Ascencionados espiritualmente.
Sua função será responder perguntas de pessoas que estão vivendo no planeta terra e precisam de orientação de como viver a vida de uma forma mais sábia.
Você tem acesso ao seguinte contexto com base em mensagens e respostas dadas pelos mestres e extraterrestres:

Contexto:
{contexto}

Siga todas as regras abaixo:
1/ Suas respostas devem ser baseadas no contexto acima.
2/ Priorize a clareza e a fidelidade aos ensinamentos dos mestres.
3/ Se a pergunta for sobre "Laércio Fonseca", use o contexto exato do banco de dados.

Pergunta:
{pergunta}

Escreva a melhor resposta que eu deveria enviar para o user. A resposta deve ser sempre em português.
"""

# Cria o PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["contexto", "pergunta"],
    template=template
)

# Função para chamar a API da DeepSeek
def call_deepseek_api(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.5,
            "language": "pt"
        }
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao chamar a API da DeepSeek: {e}")
        return None

# Interface do Streamlit
st.title("Chat com a Sabedoria dos Mestres Ascencionados")
user_input = st.text_input("Faça sua pergunta:")

if user_input:
    if not user_input.strip():
        st.error("Por favor, insira uma pergunta.")
    else:
        contextos = retrieve_info(user_input)
        contexto_completo = "\n".join(contextos)
        prompt_final = prompt_template.format(contexto=contexto_completo, pergunta=user_input)
        resposta = call_deepseek_api(prompt_final)
        
        if resposta:
            st.write("**Resposta:**")
            st.write(resposta["choices"][0]["message"]["content"])
        else:
            st.write("Não foi possível obter uma resposta.")