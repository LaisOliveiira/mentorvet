import os
import tempfile
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

from Bio import Entrez
from PIL import Image


# ============================================================
# CONFIGURAÇÕES
# ============================================================

Entrez.email = "vettutor.ifmg@gmail.com"

load_dotenv()

st.set_page_config(
    page_title="MentorVet",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# ÍCONE SVG
# ============================================================

STETHOSCOPE_SVG = """
<svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="#FFFFFF"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    width="{size}"
    height="{size}"
>
    <path d="M6 3v6a4 4 0 0 0 8 0V3"/>
    <path d="M10 15v2a4 4 0 0 0 8 0v-1a3 3 0 0 0-3-3"/>
    <circle cx="19" cy="9" r="2"/>
</svg>
"""


# ============================================================
# FUNÇÃO PARA RENDERIZAR HTML SEM VIRAR BLOCO DE CÓDIGO
# ============================================================

def render_html(html):
    st.markdown(html.strip(), unsafe_allow_html=True)


# ============================================================
# CSS — IDENTIDADE VISUAL DO MENTORVET (LIMPO E VISÍVEL)
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

:root {
    --bg: #F4F7F5;
    --surface: #FFFFFF;
    --surface-soft: #EEF2ED;
    --surface-green: #E8F2EC;
    --text: #111C16;
    --text-soft: #64746B;
    --green: #1B4D3E;
    --green-dark: #143A2E;
    --green-light: #E8F2EC;
    --border: #E2E8E4;
    --border-dark: #CBD5CD;
    --amber-badge: #D97706;
    --shadow: 0 8px 30px rgba(27, 77, 62, 0.08);
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { background: transparent !important; }

html, body, .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stApp {
    background: var(--bg) !important;
}

.block-container {
    max-width: 920px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 9rem !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.8rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 25px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
}

.sidebar-logo-icon {
    width: 42px;
    height: 42px;
    min-width: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--green) 0%, #2A735E 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(27, 77, 62, 0.2);
}

.sidebar-logo-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 17px;
    font-weight: 700;
    color: var(--text);
    line-height: 1.15;
}

.sidebar-logo-subtitle {
    margin-top: 2px;
    font-size: 11px;
    color: var(--text-soft);
}

.sidebar-section {
    margin-top: 24px;
    margin-bottom: 10px;
    font-size: 11px;
    font-weight: 700;
    color: var(--text-soft);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.sidebar-info {
    background: var(--surface-soft);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    font-size: 12px;
    line-height: 1.6;
    color: var(--text-soft);
}

.sidebar-info b {
    color: var(--text);
}

.history-item {
    font-size: 12.5px;
    font-weight: 500;
    color: var(--text);
    padding: 8px 12px;
    border-radius: 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    background: var(--bg);
    border: 1px solid var(--border);
    margin-bottom: 6px;
    transition: all 0.2s ease;
}

.history-item:hover {
    border-color: var(--green);
    background: var(--surface-green);
}

section[data-testid="stSidebar"] .stButton button,
section[data-testid="stSidebar"] .stDownloadButton button {
    width: 100%;
    min-height: 42px;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 20px -4px rgba(27, 77, 62, 0.06);
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] .stButton button:hover,
section[data-testid="stSidebar"] .stDownloadButton button:hover {
    border-color: var(--green) !important;
    background: var(--surface-green) !important;
    color: var(--green) !important;
    transform: translateY(-1px);
}

/* CABEÇALHO PRINCIPAL */
.mentor-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px 24px;
    margin-bottom: 30px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    box-shadow: 0 10px 30px -10px rgba(17, 28, 22, 0.05);
}

.mentor-icon {
    width: 52px;
    height: 52px;
    min-width: 52px;
    border-radius: 14px;
    background: linear-gradient(135deg, var(--green) 0%, #2A735E 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 6px 16px rgba(27, 77, 62, 0.25);
}

.mentor-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 24px;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
}

.mentor-subtitle {
    margin-top: 3px;
    font-size: 13px;
    color: var(--text-soft);
}

/* TELA INICIAL */
.welcome-container {
    max-width: 680px;
    margin: 70px auto 0 auto;
    text-align: center;
    padding: 45px 25px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 24px;
    box-shadow: 0 10px 30px -10px rgba(17, 28, 22, 0.05);
}

.welcome-icon {
    width: 72px;
    height: 72px;
    margin: 0 auto 20px auto;
    border-radius: 20px;
    background: linear-gradient(135deg, var(--green) 0%, #2A735E 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 8px 24px rgba(27, 77, 62, 0.25);
}

.welcome-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 30px;
    font-weight: 700;
    color: var(--text);
    line-height: 1.25;
    margin-bottom: 12px;
}

.welcome-text {
    max-width: 540px;
    margin: auto;
    font-size: 14.5px;
    line-height: 1.7;
    color: var(--text-soft);
}

/* MENSAGENS DO CHAT */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}

[data-testid="stChatMessageContent"] {
    border-radius: 18px !important;
    padding: 16px 20px !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    color: var(--text);
    border: 1px solid var(--border);
    box-shadow: 0 4px 20px -4px rgba(27, 77, 62, 0.06);
}

/* Usuário */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: var(--surface-green) !important;
    border-color: #C4DED1 !important;
    color: var(--text) !important;
}

/* Assistente */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    background: var(--surface) !important;
    border-color: var(--border) !important;
}

/* PROCESSAMENTO */
.processing-card {
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 18px;
    color: var(--text-soft);
    font-size: 13.5px;
    box-shadow: 0 4px 20px -4px rgba(27, 77, 62, 0.06);
}

.processing-dot {
    width: 9px;
    height: 9px;
    min-width: 9px;
    border-radius: 50%;
    background: var(--amber-badge);
    animation: pulse 1.2s infinite ease-in-out;
}

@keyframes pulse {
    0% { opacity: 0.3; transform: scale(0.85); }
    50% { opacity: 1; transform: scale(1.1); }
    100% { opacity: 0.3; transform: scale(0.85); }
}

/* ÁREA INFERIOR DO CHAT */
[data-testid="stBottomBlockContainer"] {
    background: transparent !important;
    border-top: none !important;
    padding-top: 12px !important;
    padding-bottom: 13px !important;
}

[data-testid="stBottomBlockContainer"] > div {
    background: transparent !important;
}

/* CHAT INPUT — VISIBILIDADE TOTAL DO TEXTO E CURSOR */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 1.5px solid var(--border-dark) !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08) !important;
    overflow: hidden !important;
    padding: 4px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 4px rgba(27, 77, 62, 0.1), 0 10px 30px rgba(0, 0, 0, 0.08) !important;
}

[data-testid="stChatInput"] > div {
    background: #FFFFFF !important;
    border: none !important;
}

[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    color: #111C16 !important;
    -webkit-text-fill-color: #111C16 !important;
    caret-color: var(--green) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14.5px !important;
    font-weight: 400 !important;
    line-height: 1.5 !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #94A3B8 !important;
    -webkit-text-fill-color: #94A3B8 !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] button {
    background-color: var(--green) !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    border: none !important;
    transition: background-color 0.2s ease !important;
}

[data-testid="stChatInput"] button:hover {
    background-color: var(--green-dark) !important;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: #CBD6CF;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #AEBDB4;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 8rem !important;
    }
    .mentor-title { font-size: 21px; }
    .mentor-subtitle { font-size: 11.5px; }
    .welcome-container { margin-top: 40px; }
    .welcome-title { font-size: 25px; }
    .welcome-text { font-size: 13.5px; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNÇÃO PUBMED
# ============================================================

def buscar_pubmed(termo, max_resultados=2):
    try:
        termo_pesquisa = f"{termo} AND (veterinary OR animal OR medicine)"
        handle = Entrez.esearch(db="pubmed", term=termo_pesquisa, retmax=max_resultados, sort="relevance")
        registro = Entrez.read(handle)
        handle.close()

        ids = registro["IdList"]
        if not ids:
            return "Nenhum artigo recente encontrado nas bases internacionais para este termo específico."

        handle_detalhes = Entrez.efetch(db="pubmed", id=ids, retmode="xml")
        artigos = Entrez.read(handle_detalhes)
        handle_detalhes.close()

        resumos_formatados = []
        for paper in artigos["PubmedArticle"]:
            try:
                titulo = paper["MedlineCitation"]["Article"]["ArticleTitle"]
                abstract_parts = paper["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", ["Resumo indisponível."])
                resumo = " ".join([str(p) for p in abstract_parts])
                resumos_formatados.append(f"Título: {titulo}\nResumo: {resumo}\n")
            except Exception:
                continue

        return "\n---\n".join(resumos_formatados)
    except Exception as e:
        return f"Erro ao acessar a base científica: {str(e)}"


# ============================================================
# CÉREBRO DO MENTORVET
# ============================================================

@st.cache_resource
def iniciar_cerebro():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    banco_base = Chroma(persist_directory="./banco_vetorial", embedding_function=embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.1)
    return embeddings, banco_base, llm


embeddings, banco_base, llm = iniciar_cerebro()


# ============================================================
# SESSION STATE
# ============================================================

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

if "historico_conversas" not in st.session_state:
    st.session_state.historico_conversas = []

if "pergunta_pendente" not in st.session_state:
    st.session_state.pergunta_pendente = None


# ============================================================
# CABEÇALHO
# ============================================================

render_html(f"""
<div class="mentor-header">
    <div class="mentor-icon">{STETHOSCOPE_SVG.format(size=24)}</div>
    <div>
        <div class="mentor-title">MentorVet</div>
        <div class="mentor-subtitle">Assistente acadêmico em Medicina Veterinária </div>
    </div>
</div>
""")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    render_html(f"""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">{STETHOSCOPE_SVG.format(size=20)}</div>
        <div>
            <div class="sidebar-logo-title">MentorVet</div>
            <div class="sidebar-logo-subtitle">Assistente acadêmico</div>
        </div>
    </div>
    """)

    st.markdown('<div class="sidebar-section">Conversa</div>', unsafe_allow_html=True)

    if st.button("＋ Nova conversa", use_container_width=True):
        if st.session_state.mensagens:
            primeira_pergunta = "Nova conversa"
            for m in st.session_state.mensagens:
                if m["role"] == "user":
                    conteudo = m["content"]
                    if isinstance(conteudo, list):
                        textos = [str(item) for item in conteudo if isinstance(item, str)]
                        if textos:
                            primeira_pergunta = textos[0]
                    else:
                        primeira_pergunta = str(conteudo)
                    break
            st.session_state.historico_conversas.insert(0, primeira_pergunta[:28] + "...")

        st.session_state.mensagens = []
        st.session_state.pergunta_pendente = None
        if "banco_aluno" in st.session_state:
            del st.session_state.banco_aluno
        st.rerun()

    if st.button("🗑️ Limpar conversa atual", use_container_width=True):
        st.session_state.mensagens = []
        st.session_state.pergunta_pendente = None
        if "banco_aluno" in st.session_state:
            del st.session_state.banco_aluno
        st.rerun()

    st.markdown('<div class="sidebar-section">Histórico (Hoje)</div>', unsafe_allow_html=True)

    if st.session_state.historico_conversas:
        for conv in st.session_state.historico_conversas[:6]:
            render_html(f"""
            <div class="history-item">
                💬 {conv}
            </div>
            """)
    else:
        render_html("""
        <div class="sidebar-info" style="margin-bottom: 10px;">
            Nenhum registro anterior.
        </div>
        """)

    if st.session_state.mensagens:
        df_historico = pd.DataFrame(st.session_state.mensagens)
        csv_data = df_historico.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Baixar histórico atual",
            data=csv_data,
            file_name="mentorvet_historico.csv",
            mime="text/csv",
            help="Baixar a conversa atual em CSV.",
            use_container_width=True
        )

    st.markdown('<div class="sidebar-section">Sobre</div>', unsafe_allow_html=True)
    render_html("""
    <div class="sidebar-info">
        <b>MentorVet</b> é um assistente acadêmico voltado para Medicina Veterinária.
        <br><br>
    </div>
    """)


# ============================================================
# TELA INICIAL
# ============================================================

if not st.session_state.mensagens:
    render_html(f"""
    <div class="welcome-container">
        <div class="welcome-icon">{STETHOSCOPE_SVG.format(size=32)}</div>
        <div class="welcome-title">Como posso ajudar nos seus estudos?</div>
        <div class="welcome-text">
            Faça uma pergunta, descreva um caso clínico ou envie um material para análise.
            <br>
            O MentorVet pode consultar sua base acadêmica e referências científicas.
        </div>
    </div>
    """)


# ============================================================
# HISTÓRICO DE MENSAGENS
# ============================================================

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if isinstance(item, str):
                    st.markdown(item)
                elif hasattr(item, "save"):
                    st.image(item, width=300)
        else:
            st.markdown(msg["content"])


# ============================================================
# INPUT PRINCIPAL (COM SUPORTE A MÚLTIPLOS ANEXOS)
# ============================================================

entrada_usuario = st.chat_input(
    "Digite sua dúvida, caso clínico ou anexe arquivos/imagens...",
    accept_file="multiple"
)

uploaded_file = None
texto_pergunta = None

if st.session_state.pergunta_pendente:
    texto_pergunta = st.session_state.pergunta_pendente
    st.session_state.pergunta_pendente = None
elif entrada_usuario:
    texto_pergunta = entrada_usuario.text if hasattr(entrada_usuario, "text") else str(entrada_usuario)
    arquivos_enviados = entrada_usuario.files if hasattr(entrada_usuario, "files") else []
    if arquivos_enviados:
        uploaded_file = arquivos_enviados[0]


# ============================================================
# PROCESSAMENTO DA SOLICITAÇÃO
# ============================================================

if texto_pergunta or uploaded_file:
    conteudo_usuario = []
    imagem_obj = None

    if not texto_pergunta:
        texto_pergunta = "Analise este arquivo ou imagem enviada."

    conteudo_usuario.append(texto_pergunta)

    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        if file_ext == "pdf":
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                documentos = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                fatias = splitter.split_documents(documentos)

                if fatias:
                    st.session_state.banco_aluno = Chroma.from_documents(fatias, embeddings)

                texto_pergunta += f"\n\n[Documento PDF anexado: {uploaded_file.name}]"
            except Exception as erro:
                texto_pergunta += f"\n\n[Erro ao processar o PDF: {erro}]"
        elif file_ext in ["png", "jpg", "jpeg"]:
            try:
                imagem_obj = Image.open(uploaded_file)
                conteudo_usuario.append(imagem_obj)
                texto_pergunta += "\n\n[Imagem/Exame visual anexado para análise]"
            except Exception as erro:
                texto_pergunta += f"\n\n[Erro ao abrir a imagem: {erro}]"
        else:
            texto_pergunta += f"\n\n[Arquivo anexado: {uploaded_file.name}]"

    st.session_state.mensagens.append({
        "role": "user",
        "content": conteudo_usuario
    })

    with st.chat_message("user"):
        st.markdown(texto_pergunta)
        if imagem_obj:
            st.image(imagem_obj, width=300)

    with st.chat_message("assistant"):
        status = st.empty()
        render_html("""
        <div class="processing-card">
            <div class="processing-dot"></div>
            <div>MentorVet está analisando sua solicitação...</div>
        </div>
        """)

        contexto_textos = []
        try:
            docs_base = banco_base.similarity_search(texto_pergunta, k=3)
            contexto_textos.extend([d.page_content for d in docs_base])
        except Exception:
            pass

        if "banco_aluno" in st.session_state:
            try:
                docs_aluno = st.session_state.banco_aluno.similarity_search(texto_pergunta, k=3)
                contexto_textos.extend([d.page_content for d in docs_aluno])
            except Exception:
                pass

        texto_combinado = "\n\n".join(contexto_textos)

        if len(texto_combinado.strip()) < 150 and not imagem_obj:
            render_html("""
            <div class="processing-card">
                <div class="processing-dot"></div>
                <div>Buscando referências científicas internacionais...</div>
            </div>
            """)

            contexto_cientifico = buscar_pubmed(texto_pergunta)
            texto_combinado += f"\n\nLiteratura Científica Recente (PubMed):\n{contexto_cientifico}"

        prompt_especialista = f"""
        Você é o MentorVet, um sistema especialista acadêmico de elite em Medicina Veterinária, sintonizado com o projeto pedagógico e a matriz curricular do Bacharelado em Medicina Veterinária(total de 3.956 horas).

        Áreas de atuação cobertas:
        - Morfologia e Fisiologia Animal
        - Sanidade e Clínica (Pequenos, Grandes, Aves, Suínos e Silvestres)
        - Zootecnia, Produção e Nutrição Animal
        - Inspeção e Tecnologia de Produtos de Origem Animal (TPOA/RIISPOA)
        - Saúde Pública, Epidemiologia e Legislação Veterinária

        Contexto extraído das bases de dados e literatura científica:
        {texto_combinado}

        Solicitação do Aluno / Descrição da Imagem:
        {texto_pergunta}

        DIRETRIZES DE COMPORTAMENTO E RESPOSTA:
        1. Análise Visual: Se o usuário enviou uma imagem (como raio-X, ultrassonografia, foto de lesão, lâmina ou exame), avalie os achados visuais sob a ótica estrita da semiologia e patologia veterinária. Não invente dados não visíveis.
        2. Modo Direto: Se o aluno pedir explicitamente uma resposta pronta, protocolo fechado, dosagem exata ou dado teórico pontual, forneça o conteúdo de forma imediata, técnica e sem rodeios.
        3. Árvore de Raciocínio Clínico / Estruturado (Obrigatório para casos abertos ou dúvidas analíticas complexas):
           - Passo 1: Identificação da área temática e dos sinais ou parâmetros centrais.
           - Passo 2: Fundamentação fisiopatológica, zootécnica, epidemiológica ou legal.
           - Passo 3: Diagnósticos diferenciais (se clínico) ou alternativas técnicas (se produção/inspeção).
           - Passo 4: Conduta recomendada baseada em evidências científicas e literaturas oficiais (como diretrizes WSAVA ou Plumb's Veterinary Drugs).
        4. Fidelidade Científica e Exclusividade: Recuse de forma direta qualquer solicitação que não envolva ciências veterinárias, agrárias ou biológicas. Não invente dados técnicos ou referências.
        """

        status.empty()

        if imagem_obj:
            try:
                resposta = llm.invoke([prompt_especialista, imagem_obj])
                resposta_final = resposta.content if hasattr(resposta, "content") else str(resposta)
                st.markdown(resposta_final)
            except Exception as erro:
                resposta_final = f"Não consegui analisar a imagem neste momento.\n\nErro: {erro}"
                st.error(resposta_final)
        else:
            try:
                cadeia = llm | StrOutputParser()
                resposta_final = st.write_stream(cadeia.stream(prompt_especialista))
            except Exception as erro:
                resposta_final = f"Não consegui gerar a resposta neste momento.\n\nErro: {erro}"
                st.error(resposta_final)

        st.session_state.mensagens.append({
            "role": "assistant",
            "content": resposta_final
        })