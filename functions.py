import requests
import pandas as pd
from io import StringIO
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from difflib import SequenceMatcher
from nltk.corpus import stopwords
import nltk

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
        print("Baixando CSV...")
        response = requests.get(google_sheets_csv_url)
        response.raise_for_status()

        print("CSV baixado com sucesso!")
        df = pd.read_csv(StringIO(response.text))

        if df.empty:
            print("Erro: O CSV está vazio.")
            return None, None

        print(f"Total de linhas no CSV: {len(df)}")

        perguntas_docs = []
        respostas_docs = []
        for _, row in df.iterrows():
            if pd.notna(row.get("Pergunta")) and pd.notna(row.get("Resposta")):
                perguntas_docs.append(Document(
                    page_content=row["Pergunta"],
                    metadata={"resposta": row["Resposta"]}
                ))
                respostas_docs.append(Document(page_content=row["Resposta"]))

        if not perguntas_docs or not respostas_docs:
            print("Erro: Nenhuma pergunta ou resposta válida foi carregada.")
            return None, None

        print(f"Total de perguntas carregadas: {len(perguntas_docs)}")
        print(f"Total de respostas carregadas: {len(respostas_docs)}")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db_perguntas = FAISS.from_documents(perguntas_docs, embeddings) if perguntas_docs else None
        db_respostas = FAISS.from_documents(respostas_docs, embeddings) if respostas_docs else None

        return db_perguntas, db_respostas

    except Exception as e:
        print(f"Erro ao carregar o CSV: {e}")
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

