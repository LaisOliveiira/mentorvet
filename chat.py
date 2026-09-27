import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Carrega as variáveis do arquivo .env (onde deve estar GROQ_API_KEY)
load_dotenv()

# 2. Carrega o mesmo modelo matemático de embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Conecta ao banco de dados vetorial local (Chroma)
vectorstore = Chroma(persist_directory="./banco_vetorial", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Configura a IA da Groq (Llama 3.3 70B - rápido, preciso e gratuito)
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.2,
    groq_api_key=groq_key
)

# 5. O PROMPT MESTRE (Guardrails)
system_prompt = (
    "Você é um tutor acadêmico especialista em Medicina Veterinária. "
    "Responda à pergunta do aluno baseando-se ÚNICA E EXCLUSIVAMENTE nos trechos de contexto fornecidos abaixo. "
    "Se a resposta não estiver no contexto, diga exatamente: 'Desculpe, não encontrei essa informação no material estudado.' "
    "NUNCA invente informações. Se o usuário perguntar sobre culinária, política, ou qualquer coisa fora da veterinária, "
    "diga: 'Sou um assistente focado apenas em estudos de Medicina Veterinária.'\n\n"
    "Contexto do livro:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 6. Função auxiliar para extrair e formatar os trechos encontrados
def formatar_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 7. Monta a cadeia RAG com a arquitetura LCEL
rag_chain = (
    {"context": retriever | formatar_documentos, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("🐾 VetTutor Iniciado! (Digite 'sair' para encerrar)")
print("Faça uma pergunta sobre o assunto do PDF que você subiu.")

# 8. Loop de conversa via terminal
while True:
    pergunta_usuario = input("\nSua pergunta: ")
    
    if pergunta_usuario.lower() == 'sair':
        print("Encerrando os estudos. Até mais!")
        break
        
    if pergunta_usuario.strip() == "":
        continue
        
    print("Buscando nos livros e raciocinando...")
    
    resposta = rag_chain.invoke(pergunta_usuario)
    
    print("\n--- RESPOSTA DO VETTUTOR ---")
    print(resposta)
    print("----------------------------")