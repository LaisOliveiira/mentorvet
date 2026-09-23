import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Carrega sua chave do Google (arquivo .env)
load_dotenv()

# 2. Carrega o mesmo modelo matemático que usamos para ler o PDF
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Conecta ao banco de dados que você criou
vectorstore = Chroma(persist_directory="./banco_vetorial", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Configura o Gemini
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)

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

# 6. Função auxiliar para extrair o texto dos parágrafos encontrados
def formatar_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 7. Monta o sistema RAG com a arquitetura moderna (LCEL)
rag_chain = (
    {"context": retriever | formatar_documentos, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("🐾 VetTutor Iniciado! (Digite 'sair' para encerrar)")
print("Faça uma pergunta sobre o assunto do PDF que você subiu.")

# 8. Loop de conversa
while True:
    pergunta_usuario = input("\nSua pergunta: ")
    
    if pergunta_usuario.lower() == 'sair':
        print("Encerrando os estudos. Até mais!")
        break
        
    if pergunta_usuario.strip() == "":
        continue
        
    print("Buscando nos livros e raciocinando...")
    
    # A IA invoca a busca, formata o texto, joga no prompt e gera a resposta
    resposta = rag_chain.invoke(pergunta_usuario)
    
    print("\n--- RESPOSTA DO VETTUTOR ---")
    print(resposta)
    print("----------------------------")