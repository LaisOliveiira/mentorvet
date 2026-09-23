import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

print("Consultando os servidores do Google...")
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

try:
    response = requests.get(url)
    dados = response.json()
    
    if "error" in dados:
        print("Erro na chave da API:", dados["error"]["message"])
    else:
        print("\n=== MODELOS DE CHAT LIBERADOS PARA SUA CONTA ===")
        for modelo in dados.get("models", []):
            nome = modelo.get("name", "").replace("models/", "")
            metodos = modelo.get("supportedGenerationMethods", [])
            
            # Filtra apenas as IAs que servem para conversar (gerar texto)
            if "generateContent" in metodos:
                print(f"-> {nome}")
                
except Exception as e:
    print(f"Erro de conexão: {e}")