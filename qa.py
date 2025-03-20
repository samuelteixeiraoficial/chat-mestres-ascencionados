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
import langid  # Para detecção de idioma

# Carrega as variáveis de ambiente
load_dotenv()

# Link público do Google Sheets (exportado como CSV)
google_sheets_csv_url = "https://docs.google.com/spreadsheets/d/1E0xHCuPXFx6TR8CgiVZvD37KizSsljT9D7eTd8lA9Aw/export?format=csv"

# Carrega o CSV diretamente do link
try:
    # Faz o download do CSV
    response = requests.get(google_sheets_csv_url)
    response.raise_for_status()  # Verifica se a requisição foi bem-sucedida

    # Usa o pandas para ler o CSV
    df = pd.read_csv(StringIO(response.text))

    # Converte o DataFrame para uma lista de objetos Document (formato esperado pelo LangChain)
    documents = [Document(page_content=row.to_string())
                 for _, row in df.iterrows()]
except Exception as e:
    st.error(f"Erro ao carregar o CSV: {e}")

# Configuração dos embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

# Cria o banco de dados de vetores com FAISS
db = FAISS.from_documents(documents, embeddings)

# Função para buscar documentos semelhantes


def retrieve_info(query):
    # Busca os 7 documentos mais semelhantes
    similar_response = db.similarity_search(query, k=7)
    return [doc.page_content for doc in similar_response]


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

4/ Algumas questões e perguntas do user não são respondidas diretamente à ele, pois a reflexão e meditação deve fazer parte da jornada de aprendizado do user.

5/ Metáforas podem ser criadas por você com base no contexto para que o user tenha a possibilidade de refletir de forma profunda com base em um conhecimento e uma linha de raciocínio de uma resposta que o user não esperaria receber.

6/ Algumas respostas podem ser curtas, outras podem ser longas exatamente como nos modelos do contexto, pois o mais importante é falar o necessário e apenas o necessário.

7/ No contexto pode possuir interação de respostas dos discípulos em relação a interrogações que podem ser feitas pelos mestres durante suas próprias respostas, para que o aprendizado seja mais profundo eles mantêm essa interação na conversa entre mestre e discípulo.

8/ Considere que o contexto é um banco de dados de conversas que aconteceram em vários dias, onde cada dia pode ter tido 2, 3, 5, 10, etc. perguntas e respostas, inclusive onde uma resposta do mestre poderia levar para a próxima questão a seguir no mesmo dia.

Pergunta:
{pergunta}

Idioma da Resposta:
{idioma}

Escreva a melhor resposta que eu deveria enviar para o user.
"""

# Template alternativo para quando não há contexto relevante
template_sem_contexto = """
Você é um assistente virtual de uma egrégora de seres Ascencionados espiritualmente.
Sua função será responder perguntas de pessoas que estão vivendo no planeta terra e precisam de orientação de como viver a vida de uma forma mais sábia.

A pergunta do usuário não tem relação direta com o contexto das mensagens e respostas dos mestres ascencionados e extraterrestres. Portanto, siga as regras abaixo:

1/ Informe ao usuário que a resposta não tem base no conteúdo real do banco de dados das respostas dos mestres ascencionados e extraterrestres.

2/ Dê uma resposta simples e curta, mas que ainda seja útil e relevante para a pergunta do usuário.

Pergunta:
{pergunta}

Idioma da Resposta:
{idioma}

Escreva a melhor resposta que eu deveria enviar para o user.
"""

# Cria os PromptTemplates
prompt_template = PromptTemplate(
    input_variables=["contexto", "pergunta", "idioma"],
    template=template
)

prompt_template_sem_contexto = PromptTemplate(
    input_variables=["pergunta", "idioma"],
    template=template_sem_contexto
)

# Função para chamar a API da DeepSeek


def call_deepseek_api(prompt, idioma):
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",  # Substitua pelo modelo correto, se necessário
            "messages": [
                {
                    "role": "user",  # Papel do usuário
                    "content": prompt  # Conteúdo da mensagem
                }
            ],
            "max_tokens": 500,
            "temperature": 0.5,
            "language": idioma  # Define o idioma da resposta
        }
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        response.raise_for_status()  # Levanta uma exceção para erros HTTP
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao chamar a API da DeepSeek: {e}")
        return None


# Interface do Streamlit
st.title("Chat com a Sabedoria dos Mestres Ascencionados")

# Campo de entrada para a pergunta
user_input = st.text_input("Faça sua pergunta:")

if user_input:
    if not user_input.strip():
        st.error("Por favor, insira uma pergunta.")
    else:
        # Detecta o idioma da pergunta
        try:
            idioma, _ = langid.classify(user_input)
        except:
            # Define um idioma padrão (português) caso a detecção falhe
            idioma = "pt"

        # Busca documentos semelhantes
        contextos = retrieve_info(user_input)

        # Verifica se há contexto relevante
        if not contextos or all(not contexto.strip() for contexto in contextos):
            # Se não houver contexto relevante, usa o template alternativo
            prompt_final = prompt_template_sem_contexto.format(
                pergunta=user_input,
                idioma=idioma
            )
        else:
            # Combina os contextos em um único texto
            contexto_completo = "\n".join(contextos)

            # Cria o prompt com o template principal
            prompt_final = prompt_template.format(
                contexto=contexto_completo,
                pergunta=user_input,
                idioma=idioma
            )

        # Chama a API da DeepSeek
        resposta = call_deepseek_api(prompt_final, idioma)
        if resposta:
            # Exibe apenas a resposta
            st.write("**Resposta:**")
            st.write(resposta["choices"][0]["message"]["content"])
        else:
            st.write("Não foi possível obter uma resposta.")
