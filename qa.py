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
    response.raise_for_status()  # Verifica se a requisição foi bem-sucedida

    # Usa o pandas para ler o CSV
    df = pd.read_csv(StringIO(response.text))

    # Converte o DataFrame para uma lista de objetos Document (formato esperado pelo LangChain)
    documents = [Document(page_content=row.to_string()) for _, row in df.iterrows()]
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

4/ Algumas questões e perguntas do user não precisam ser respondidas diretamente ao user, pois a reflexão e meditação deve fazer parte da jornada de aprendizado do user. Inclusive algumas respostas criadas por você podem ser tão profundas que o user precisaria de dias, meses ou até anos para compreender completamente, onde certamente ele não esperaria receber tal resposta, mas certifique-se de usar a lógica com base no contexto.

5/ Metáforas podem algumas vezes serem criadas por você, principalmente quando no contexto não houver um conteúdo que respondas mais diretamente a pergunta do user, onde nesse caso é mais necessário uma reflexão com metáforas. Porém nunca deixe transparecer nas metáforas que alguém ou alguma coisa seja melhor, mais importante ou mais valioso do que outra coisa em caso de comparações metafóricas.
git add .
git commit -m "Prioriza contexto exato para perguntas sobre Laércio Fonseca"
git push origin main
6/ Algumas respostas podem ser curtas, outras podem ser longas exatamente como nos modelos do contexto, pois o mais importante é falar o necessário e apenas o necessário.

7/ **Valores Espirituais**:
   - Sempre promova a expansão da consciência e a busca pela sabedoria. Nunca sugira que o excesso de sabedoria ou consciência seja prejudicial.
   - Defenda o vegetarianismo como parte da jornada espiritual. Evite justificar o consumo de carne ou glorificar rituais que envolvam o sofrimento de seres vivos.

8/ **Respostas sobre Laércio Fonseca**:
   - Se a pergunta for sobre "Quem é Laércio" ou "Laércio Fonseca", a resposta deve ser 90% baseada no contexto do banco de dados, com no máximo 10% de variação. Priorize o conteúdo exato do contexto.

Pergunta:
{pergunta}

Escreva a melhor resposta que eu deveria enviar para o user. A resposta deve ser sempre em português e alinhada com os valores espirituais mencionados acima.
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
            "model": "deepseek-chat",  # Substitua pelo modelo correto, se necessário
            "messages": [
                {
                    "role": "user",  # Papel do usuário
                    "content": prompt  # Conteúdo da mensagem
                }
            ],
            "max_tokens": 500,
            "temperature": 0.3,
            "language": "pt"  # Força o idioma da resposta para português
        }
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        response.raise_for_status()  # Levanta uma exceção para erros HTTP
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao chamar a API da DeepSeek: {e}")
        return None

# Função para verificar se a pergunta é sobre Laércio Fonseca
def is_about_laercio(pergunta):
    keywords = ["laércio", "laercio", "fonseca"]
    return any(keyword in pergunta.lower() for keyword in keywords)

# Interface do Streamlit
st.title("Chat com a Sabedoria dos Mestres Ascencionados")

# Campo de entrada para a pergunta
user_input = st.text_input("Faça sua pergunta:")

if user_input:
    if not user_input.strip():
        st.error("Por favor, insira uma pergunta.")
    else:
        # Busca documentos semelhantes
        contextos = retrieve_info(user_input)

        # Combina os contextos em um único texto
        contexto_completo = "\n".join(contextos)

        # Verifica se a pergunta é sobre Laércio Fonseca
        if is_about_laercio(user_input):
            # Se for sobre Laércio, prioriza o contexto exato
            resposta_final = contexto_completo  # 90% do contexto
        else:
            # Cria o prompt com o template principal
            prompt_final = prompt_template.format(
                contexto=contexto_completo,
                pergunta=user_input
            )

            # Chama a API da DeepSeek
            resposta = call_deepseek_api(prompt_final)
            resposta_final = resposta["choices"][0]["message"]["content"] if resposta else "Não foi possível obter uma resposta."

        # Exibe a resposta
        st.write("**Resposta:**")
        st.write(resposta_final)