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


# Função para carregar dados do Google Sheets
def carregar_dados(google_sheets_csv_url):
    try:
        response = requests.get(google_sheets_csv_url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))

        perguntas_docs = []
        respostas_docs = []
        for _, row in df.iterrows():
            if pd.notna(row["Pergunta"]) and pd.notna(row["Resposta"]):
                perguntas_docs.append(Document(
                    page_content=row["Pergunta"],
                    metadata={"resposta": row["Resposta"]}
                ))
                respostas_docs.append(Document(
                    page_content=row["Resposta"]
                ))
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        db_perguntas = FAISS.from_documents(perguntas_docs, embeddings)
        db_respostas = FAISS.from_documents(respostas_docs, embeddings)
        
        return db_perguntas, db_respostas
        
    except Exception as e:
        raise Exception(f"Erro ao carregar o CSV: {e}")
    
def calcular_similaridade(pergunta, perguntas_banco):
    """
    Calcula a similaridade entre a pergunta do usuário e as perguntas do banco de dados.
    Retorna o índice da pergunta mais similar e o valor da similaridade.
    """
    vectorizer = TfidfVectorizer(stop_words='portuguese')
    todas_perguntas = perguntas_banco + [pergunta]  # Adiciona a pergunta do usuário

    tfidf_matrix = vectorizer.fit_transform(todas_perguntas)  # Vetoriza as perguntas
    similaridade = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])  # Similaridade com as perguntas do banco

    indice_max = np.argmax(similaridade)  # Índice da pergunta mais similar
    similaridade_max = similaridade[0][indice_max]  # Valor da maior similaridade
    
    return indice_max, similaridade_max


# Função para carregar o template
def carregar_template(template_path):
    try:
        with open(template_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        raise Exception(f"Erro ao carregar o template: {e}")

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
        raise Exception(f"Erro no processamento: {str(e)}")

