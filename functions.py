import requests
import pandas as pd
from io import StringIO
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import os

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

# Função para carregar o template
def carregar_template(template_path):
    try:
        with open(template_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        raise Exception(f"Erro ao carregar o template: {e}")

# Função para processar a pergunta
def processar_pergunta(pergunta, db_perguntas, db_respostas, template, api_key):
    try:
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
        
        return resposta["choices"][0]["message"]["content"]
    
    except Exception as e:
        raise Exception(f"Erro no processamento: {str(e)}")
