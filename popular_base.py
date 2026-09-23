import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Coloque a sua chave real aqui
os.environ['PINECONE_API_KEY'] = 'pcsk_4wxRZ7_BmYg6AxjJCSSmBmjAqP7x5mipWZNQRiYH8PnHK6Eg2PosBBCMyXG46J7XzoyoA8'

print("Lendo todos os livros da pasta de uma vez...")
# Esse Loader pega automaticamente todos os PDFs dentro da pasta 'livros'
loader = PyPDFDirectoryLoader("livros")
paginas = loader.load()

print(f"Foram lidas {len(paginas)} páginas no total. Fatiando tudo...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
trechos = text_splitter.split_documents(paginas)

print("Gerando vetores e enviando para o Pinecone (vá tomar um café, isso vai demorar)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

banco_nuvem = PineconeVectorStore.from_documents(
    documents=trechos, 
    embedding=embeddings, 
    index_name="mentorvet"
)

print("🎉 SUCESSO ABSOLUTO! Toda a sua biblioteca médica está na nuvem!")