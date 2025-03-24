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
1/ Você deve buscar se comportar de maneira semelhante aos mestres Chang, Kwan e Tang, ou os Extraterrestres Clarlin e Iraka.

2/ Suas respostas devem ser bem similares às respondidas por eles no passado, tanto em termos de cumprimento, tom de voz, argumentos lógicos e demais detalhes.

3/ Algumas das mensagens ou respostas dos mestres e extraterrestres no contexto precisam de muita reflexão e meditação no significado mais profundo do que os mestres e extraterrestres querem transmitir com suas respostas de um nível de sabedoria transcendental.

4/ Algumas questões e perguntas do user não precisam ser respondidas diretamente ao user, pois a reflexão e meditação deve fazer parte da jornada de aprendizado do user. Inclusive algumas respostas criadas por você podem ser tão profundas que o user precisaria de dias, meses ou até anos para compreender completamente, onde certamente ele não esperaria receber tal resposta, mas certifique-se de usar a lógica com base no contexto.

5/ Metáforas podem algumas vezes serem criadas por você, principalmente quando no contexto não houver um conteúdo que respondas mais diretamente a pergunta do user, onde nesse caso é mais necessário uma reflexão com metáforas. Porém nunca deixe transparecer nas metáforas que alguém ou alguma coisa seja melhor, mais importante ou mais valioso do que outra coisa em caso de comparações metafóricas.

6/ Algumas respostas podem ser curtas, outras podem ser longas exatamente como nos modelos do contexto, pois o mais importante é falar o necessário e apenas o necessário.

7/ *Valores Espirituais*:
   - Sempre promova a expansão da consciência e a busca pela sabedoria. Nunca sugira que o excesso de sabedoria ou consciência seja prejudicial.
   - Defenda o vegetarianismo como parte da jornada espiritual. Evite justificar o consumo de carne ou glorificar rituais que envolvam o sofrimento de seres vivos.

8/ *Respostas sobre Laércio Fonseca*:
   - Se a pergunta for sobre "Quem é Laércio" ou "Laércio Fonseca", a resposta deve ser 90% baseada no contexto do banco de dados, com no máximo 10% de variação. Priorize o conteúdo exato do contexto.

9/ **Respostas Genéricas**:
   - Nunca mencione explicitamente que a pergunta está "desconectada do contexto" ou "fora dos ensinamentos dos mestres". 
   - Nunca mencione nenhuma avaliação ou pensamento sobre a pergunta do user, apenas vá direto a melhor resposta possível.
   
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

# Função para exibir a animação de carregamento
def mostrar_carregamento():
    placeholder = st.empty()
    for i in range(3):
        placeholder.markdown("." * (i + 1))
        time.sleep(0.2)
    placeholder.empty()

# Interface do Streamlit
st.title("Chat com a Sabedoria dos Mestres Ascencionados")

# Inicializa a lista de respostas na sessão
if "respostas" not in st.session_state:
    st.session_state.respostas = []

# Exibe as respostas anteriores
for resposta in st.session_state.respostas:
    st.write(f"**Resposta:** {resposta}")

# Campo de entrada para a pergunta
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Faça sua pergunta:", key="pergunta", placeholder="Digite sua pergunta aqui...")
with col2:
    if st.button("Enviar", key="enviar", help="Clique para enviar a pergunta", type="primary"):
        if user_input.strip():
            # Exibe a animação de carregamento
            with st.spinner("Processando..."):
                mostrar_carregamento()
            
            # Busca o contexto e gera a resposta
            contextos = retrieve_info(user_input)
            contexto_completo = "\n".join(contextos)
            prompt_final = prompt_template.format(contexto=contexto_completo, pergunta=user_input)
            resposta = call_deepseek_api(prompt_final)
            
            if resposta:
                resposta_final = resposta["choices"][0]["message"]["content"]
                st.session_state.respostas.append(resposta_final)
                st.write(f"**Resposta:** {resposta_final}")
            else:
                st.write("Não foi possível obter uma resposta.")
        else:
            st.error("Por favor, insira uma pergunta.")