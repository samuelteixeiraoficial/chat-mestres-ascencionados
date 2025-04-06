# 📦 Bibliotecas padrão
import os
import logging
from io import StringIO

# 📥 Requisições HTTP e manipulação de dados
import requests
import pandas as pd
import numpy as np

# 🧠 NLP e similaridade
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords

# 🔗 LangChain e vetores
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

logging.basicConfig(level=logging.INFO)

# Baixando as stopwords em português, se necessário
nltk.download('stopwords')

# Carregar as stopwords em português
stop_words_pt = stopwords.words('portuguese')

# Agora, crie o TfidfVectorizer com as stopwords em português
vectorizer = TfidfVectorizer(stop_words=stop_words_pt)

def calcular_similaridade(pergunta, perguntas_banco):
    """
    Calcula a similaridade entre a pergunta do usuário e as perguntas do banco de dados.
    Retorna o índice da pergunta mais similar e o valor da similaridade.
    """
    todas_perguntas = perguntas_banco + [pergunta]  # Adiciona a pergunta do usuário

    tfidf_matrix = vectorizer.fit_transform(todas_perguntas)  # Vetoriza as perguntas
    similaridade = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])  # Similaridade com as perguntas do banco

    indice_max = np.argmax(similaridade)  # Índice da pergunta mais similar
    similaridade_max = similaridade[0][indice_max]  # Valor da maior similaridade
    
    return indice_max, similaridade_max


# Função para carregar dados do Google Sheets
def carregar_dados(google_sheets_csv_url):
    try:
        print("📥 Baixando CSV...")
        response = requests.get(google_sheets_csv_url)
        response.raise_for_status()

        csv_text = response.content.decode("utf-8-sig", errors="replace")

        # 🔍 Mostrar os primeiros 1000 caracteres do CSV bruto nos logs
        logging.info("📄 Conteúdo bruto do CSV:\n%s", csv_text[:1000])

        # Tenta carregar o DataFrame
        df = pd.read_csv(StringIO(csv_text), sep=None, engine="python")

        print("✅ CSV carregado como DataFrame com sucesso!")
        print("📋 Colunas detectadas:", df.columns.tolist())
        logging.info("📋 Colunas detectadas: %s", df.columns.tolist())

        # Normaliza os nomes das colunas
        def normalizar_nome(col):
            return col.strip().lower()

        colunas_norm = {normalizar_nome(col): col for col in df.columns}

        if "pergunta" not in colunas_norm or "resposta" not in colunas_norm:
            print("🚫 Erro: Colunas 'Pergunta' e/ou 'Resposta' não foram encontradas no CSV.")
            print("📋 Colunas disponíveis:", df.columns.tolist())
            return None, None

        col_pergunta = colunas_norm["pergunta"]
        col_resposta = colunas_norm["resposta"]

        # Cria os documentos
        perguntas_docs = []
        respostas_docs = []
        for _, row in df.iterrows():
            pergunta = row[col_pergunta]
            resposta = row[col_resposta]
            if pd.notna(pergunta) and pd.notna(resposta):
                perguntas_docs.append(Document(page_content=pergunta, metadata={"resposta": resposta}))
                respostas_docs.append(Document(page_content=resposta))

        if not perguntas_docs or not respostas_docs:
            print("⚠️ Erro: Nenhuma pergunta ou resposta válida foi carregada.")
            return None, None

        print(f"📦 Total de perguntas carregadas: {len(perguntas_docs)}")
        print(f"📦 Total de respostas carregadas: {len(respostas_docs)}")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db_perguntas = FAISS.from_documents(perguntas_docs, embeddings)
        db_respostas = FAISS.from_documents(respostas_docs, embeddings)

        print("✅ Dados carregados e bancos vetoriais criados com sucesso!")
        return db_perguntas, db_respostas

    except Exception as e:
        print(f"❌ Erro ao carregar os dados do banco: {e}")
        return None, None


# Função para carregar o template
def carregar_template(template_path):
    try:
        with open(template_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        raise Exception(f"Erro ao carregar o template: {e}")


def verificar_dados(db_perguntas, db_respostas):
    if db_perguntas is None or db_respostas is None:
        raise Exception("Erro: O banco de dados não foi carregado corretamente.")


def processar_pergunta(pergunta, db_perguntas, db_respostas, template, api_key):
    try:
        # Obtém todas as perguntas do banco
        perguntas_lista = [doc.page_content for doc in db_perguntas.similarity_search("", k=100)]  # Pega até 100 perguntas armazenadas

        # Calcula a similaridade da pergunta do usuário com as do banco
        indice_similar, similaridade_pergunta = calcular_similaridade(pergunta, perguntas_lista)

        if similaridade_pergunta >= 0.70:  # Se for ≥ 70% similar
            resposta_banco = db_respostas.similarity_search(perguntas_lista[indice_similar], k=1)[0].page_content

            # Verifica a similaridade da resposta gerada
            similaridade_resposta = SequenceMatcher(None, resposta_banco, template).ratio()

            if similaridade_resposta < 0.80:  # Se a resposta gerada for muito diferente
                resposta_final = f"{resposta_banco} (Baseado no conhecimento registrado)"
            else:
                resposta_final = resposta_banco

        else:
            # Fluxo normal se não for similar a nenhuma pergunta no banco
            similar_perguntas = db_perguntas.similarity_search_with_score(pergunta, k=4)
            usar_respostas = all(score < 0.2 for _, score in similar_perguntas)

            if usar_respostas:
                contextos = [doc.page_content for doc in db_respostas.similarity_search(pergunta, k=7)]
            else:
                contextos = [doc.metadata["resposta"] for doc, _ in similar_perguntas]

            prompt = PromptTemplate(
                template=template,
                input_variables=["contexto", "pergunta"]
            ).format(
                contexto="\n".join(contextos),
                pergunta=pergunta
            )

            headers = {"Authorization": f"Bearer {api_key}"}
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

            resposta_final = resposta["choices"][0]["message"]["content"]

        return resposta_final

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

