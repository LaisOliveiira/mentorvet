import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Importação nova usando HuggingFace:
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def processar_pdf(caminho_pdf):
    print(f"Lendo o arquivo: {caminho_pdf}...")
    
    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()

    print("Fatiando o texto para a IA...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    textos_fatiados = text_splitter.split_documents(documentos)
    print(f"O PDF foi dividido em {len(textos_fatiados)} pedaços.")

    # NOVO: Gerando os vetores no seu próprio computador usando um modelo leve e rápido
    print("Baixando modelo local e gerando coordenadas matemáticas...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=textos_fatiados,
        embedding=embeddings,
        persist_directory="./banco_vetorial"
    )
    print("Sucesso! O livro agora faz parte do conhecimento do seu VetTutor.")

if __name__ == "__main__":
    arquivo_alvo = "dados/livro_teste.pdf"
    
    if os.path.exists(arquivo_alvo):
        processar_pdf(arquivo_alvo)
    else:
        print("ERRO: Não encontrei o arquivo. Coloque um PDF na pasta 'dados' com o nome 'livro_teste.pdf'.")