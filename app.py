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
from langchain_pinecone import PineconeVectorStore

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
# CSS PREMIUM — MENTORVET / CHAT MODERNO
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600;650;700&display=swap');

:root {
    --bg: #ffffff;
    --surface: #ffffff;
    --soft: #f7f7f8;
    --text: #1f1f1f;
    --muted: #6b7280;
    --line: #e5e7eb;
    --green: #117a5b;
    --green-2: #0e684e;
    --user: #f1f6f3;
}

* { box-sizing: border-box; }
html, body, .stApp {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    color: var(--text) !important;
}
body, .stApp { background: var(--bg) !important; }
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }

/* Esconde só o botão "Deploy" e o indicador de status, sem matar
   a toolbar inteira — escondê-la inteira também apagava o botão
   de abrir/fechar a sidebar, que provavelmente vive no mesmo contêiner. */
[data-testid="stStatusWidget"] { visibility: hidden !important; }
[data-testid="stAppDeployButton"] { display: none !important; }
.stAppDeployButton { display: none !important; }
button[title="Deploy this app"] { display: none !important; }

/* Remove o aspecto de página Streamlit */
.block-container {
    max-width: 100% !important;
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    padding-bottom: 8.5rem !important;
}

/* ------------------------------------------------------------
   Marca no topo do painel principal — agora fixa no FLUXO
   normal do documento (sticky), não mais "position: fixed".
   Isso corrige a sobreposição com o nome na sidebar.
------------------------------------------------------------- */
.app-brand {
    position: sticky;
    top: 0;
    z-index: 30;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    max-width: 820px;
    margin: 0 auto;
    padding: 13px 20px;
    background: rgba(255,255,255,.94);
    backdrop-filter: blur(6px);
    border-bottom: 1px solid var(--line);
}
.app-brand-left { display: flex; align-items: center; gap: 10px; }
.app-brand-tag { font-size: 11px; color: var(--muted); }
.brand-mark, .sidebar-brand-mark, .assistant-mark, .welcome-mark {
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--green);
    flex-shrink: 0;
}
.brand-mark { width: 30px; height: 30px; border-radius: 9px; }
.brand-name { font-size: 14px; font-weight: 650; line-height: 1.1; }
.brand-desc { font-size: 11px; color: var(--muted); margin-top: 2px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #f7f7f8 !important;
    border-right: 1px solid #e7e7e8 !important;
}
section[data-testid="stSidebar"] > div {
    padding: 14px 13px 18px !important;
}
.sidebar-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 5px 8px 17px;
    margin-bottom: 13px;
    border-bottom: 1px solid #e6e6e7;
}
.sidebar-brand-mark { width: 32px; height: 32px; border-radius: 9px; }
.sidebar-brand-name { font-size: 14px; font-weight: 650; }
.sidebar-brand-desc { font-size: 10.5px; color: var(--muted); margin-top: 2px; }
.sidebar-label {
    font-size: 10px; text-transform: uppercase; letter-spacing: .07em;
    color: #8a8d91; font-weight: 650; padding: 12px 9px 8px;
}
.sidebar-empty { font-size: 12px; color: #96999d; padding: 8px 9px; line-height: 1.5; }
.history-row {
    display: flex; align-items: center; gap: 8px;
    padding: 9px 9px; margin: 2px 0; border-radius: 9px;
    font-size: 12px; color: #3e4145; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
}
.history-row:hover { background: #ececed; }
.history-dot { color: #a2a5a8; font-size: 15px; line-height: 0; }
.sidebar-spacer { height: 8px; }
.sidebar-footer {
    border-top: 1px solid #e6e6e7; margin-top: 13px; padding: 15px 9px 0;
    font-size: 10.5px; line-height: 1.55; color: #8a8d91;
}
.sidebar-footer-title { color: #45484c; font-weight: 650; margin-bottom: 2px; }

/* Botões nativos da sidebar */
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] .stDownloadButton > button {
    width: 100% !important;
    min-height: 39px !important;
    border: 1px solid #dedfe1 !important;
    border-radius: 9px !important;
    background: #ffffff !important;
    color: #34373a !important;
    box-shadow: none !important;
    font-size: 12.5px !important;
    font-weight: 550 !important;
    justify-content: flex-start !important;
    gap: 8px !important;
    transition: .15s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover,
section[data-testid="stSidebar"] .stDownloadButton > button:hover {
    border-color: #c7d8d1 !important;
    background: #f0f6f3 !important;
    color: var(--green) !important;
}

/* ------------------------------------------------------------
   Botão de abrir/fechar a sidebar.
   Confirmado via DevTools: o ícone é um <span data-testid="stIconMaterial">
   dentro de um botão cujo container estável (documentado, não muda
   entre versões) é div[data-testid="collapsedControl"]. Miramos nisso
   diretamente, com fundo verde sólido — nunca transparente/branco.
------------------------------------------------------------- */
div[data-testid="collapsedControl"] {
    z-index: 999999 !important;
}
div[data-testid="collapsedControl"] button,
div[data-testid="collapsedControl"] [role="button"] {
    background-color: var(--green) !important;
    border-radius: 10px !important;
    width: 38px !important;
    height: 38px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,.14) !important;
}
div[data-testid="collapsedControl"] button:hover,
div[data-testid="collapsedControl"] [role="button"]:hover {
    background-color: var(--green-2) !important;
}
div[data-testid="collapsedControl"] [data-testid="stIconMaterial"] {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* Fallback com correspondência parcial, para o caso de o botão de
   FECHAR (sidebar já aberta) usar um testid diferente do de abrir. */
button[aria-label*="idebar" i],
button[data-testid*="idebar" i],
button[data-testid*="ollaps" i] {
    background-color: var(--green) !important;
    border-radius: 10px !important;
}
button[aria-label*="idebar" i]:hover,
button[data-testid*="idebar" i]:hover,
button[data-testid*="ollaps" i]:hover {
    background-color: var(--green-2) !important;
}
button[aria-label*="idebar" i] [data-testid="stIconMaterial"],
button[data-testid*="idebar" i] [data-testid="stIconMaterial"],
button[data-testid*="ollaps" i] [data-testid="stIconMaterial"] {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* ------------------------------------------------------------
   Tela inicial — título + sugestões dentro de UM MESMO
   container real (st.container(key=...)), centralizados juntos.
------------------------------------------------------------- */
.st-key-welcome_wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    max-width: 720px;
    margin: 0 auto;
    min-height: calc(100vh - 230px);
    padding: 24px 24px 10px;
}
.welcome-mark {
    width: 56px; height: 56px; border-radius: 17px; margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(17,122,91,.16);
}
.st-key-welcome_wrap h1 {
    margin: 0 0 10px !important;
    font-size: clamp(26px, 4vw, 34px) !important;
    letter-spacing: -.04em !important;
    font-weight: 650 !important;
    color: #202124 !important;
}
.st-key-welcome_wrap p {
    max-width: 560px; margin: 0 !important;
    color: #73777c; font-size: 14px; line-height: 1.7;
}
.st-key-welcome_wrap [data-testid="stHorizontalBlock"] {
    width: 100%;
    margin-top: 30px;
    gap: 10px !important;
}
.st-key-welcome_wrap .stButton button {
    width: 100%;
    text-align: left;
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 13px 15px;
    color: #2c2f31;
    font-size: 12.5px;
    line-height: 1.45;
    white-space: normal;
    height: auto;
    box-shadow: none;
    justify-content: flex-start;
    gap: 9px;
    transition: border-color .15s ease, transform .1s ease, background .15s ease;
}
.st-key-welcome_wrap .stButton button:hover {
    border-color: var(--green);
    background: #f4faf7;
    color: var(--green-2);
    transform: translateY(-1px);
}

/* Área de conversa */
.st-key-history_column, .st-key-current_column {
    max-width: 820px;
    margin: 0 auto;
}
.message-row {
    width: 100%;
}
/* Fix estrutural: o texto de resposta (st.markdown / st.write_stream) nunca
   tinha nenhum limite de largura próprio — só o cabeçalho "MentorVet" tinha.
   Aplicamos a régua de 820px direto neste seletor, que já sabemos que
   funciona (é o mesmo usado pela tipografia customizada abaixo), e o
   escopamos a ".block-container" para não vazar para dentro da sidebar. */
.block-container [data-testid="stMarkdownContainer"] {
    max-width: 820px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: 24px !important;
    padding-right: 24px !important;
    box-sizing: border-box !important;
}
.user-row { display: flex; justify-content: flex-end; margin-top: 25px; margin-bottom: 30px; }
.user-bubble {
    max-width: min(620px, 78%);
    background: var(--user);
    border: 1px solid #e3ebe7;
    border-radius: 20px;
    padding: 11px 16px;
    font-size: 14px;
    line-height: 1.55;
    color: #242725;
}
.assistant-row { margin-top: 22px; margin-bottom: 7px; }
.assistant-head {
    display: flex; align-items: center; gap: 9px;
    font-size: 13px; font-weight: 650; color: #25282a;
    margin-bottom: 11px;
}
.assistant-mark {
    width: 25px; height: 25px; border-radius: 8px;
}

/* Remove caixas padrão de chat e melhora Markdown */
[data-testid="stChatMessage"] { background: transparent !important; border: 0 !important; padding: 0 !important; }
[data-testid="stChatMessageContent"] {
    background: transparent !important; border: 0 !important; box-shadow: none !important;
    padding: 0 !important; border-radius: 0 !important;
}

.stMarkdown, [data-testid="stMarkdownContainer"] {
    font-size: 15px !important;
    line-height: 1.78 !important;
}
[data-testid="stMarkdownContainer"] p { margin: 0 0 17px !important; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #202224 !important; letter-spacing: -.025em !important;
    margin-top: 27px !important; margin-bottom: 12px !important;
}
[data-testid="stMarkdownContainer"] h2 { font-size: 19px !important; }
[data-testid="stMarkdownContainer"] h3 { font-size: 16px !important; }
[data-testid="stMarkdownContainer"] ul, [data-testid="stMarkdownContainer"] ol { margin-bottom: 18px !important; }
[data-testid="stMarkdownContainer"] li { margin: 5px 0 !important; }
[data-testid="stMarkdownContainer"] strong { font-weight: 650 !important; }
[data-testid="stMarkdownContainer"] blockquote {
    border-left: 3px solid #b8cec4; margin: 18px 0; padding: 4px 0 4px 15px; color: #5f6662;
}
[data-testid="stMarkdownContainer"] code {
    background: #f3f4f5; border: 1px solid #e6e7e8; border-radius: 5px; padding: 2px 5px;
}
[data-testid="stMarkdownContainer"] pre {
    border-radius: 10px !important; border: 1px solid #e5e6e7 !important;
}

/* Tabelas */
[data-testid="stMarkdownContainer"] table { border-collapse: collapse; width: 100%; margin: 16px 0 22px; }
[data-testid="stMarkdownContainer"] th { background: #f7f7f7; font-weight: 650; }
[data-testid="stMarkdownContainer"] th, [data-testid="stMarkdownContainer"] td {
    border: 1px solid #e5e5e5; padding: 9px 11px; text-align: left;
}

/* Loading minimalista */
.loading-line { display:flex; align-items:center; gap:5px; color:#8a8f93; font-size:12px; margin: 2px 0 22px 34px; }
.loading-dot { width:5px; height:5px; border-radius:50%; background:#9eb4aa; animation: blink 1.2s infinite ease-in-out; }
.loading-dot:nth-child(2){ animation-delay:.15s; }
.loading-dot:nth-child(3){ animation-delay:.30s; }
@keyframes blink { 0%,80%,100%{opacity:.25; transform:translateY(0)} 40%{opacity:1; transform:translateY(-2px)} }

/* Input inferior — referência ChatGPT/Gemini */
[data-testid="stBottomBlockContainer"] {
    background: linear-gradient(to top, #fff 65%, rgba(255,255,255,.96) 100%) !important;
    border-top: 0 !important;
    padding: 14px 0 15px !important;
}
[data-testid="stBottomBlockContainer"] > div {
    max-width: 820px !important;
    margin: 0 auto !important;
    background: transparent !important;
}
[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 1px solid #d8dadd !important;
    border-radius: 24px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,.06) !important;
    padding: 5px 7px 5px 11px !important;
    overflow: hidden !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #bfc5c2 !important;
    box-shadow: 0 3px 16px rgba(0,0,0,.08) !important;
}
[data-testid="stChatInput"] > div { background: transparent !important; border: 0 !important; }
[data-testid="stChatInput"] textarea {
    background: transparent !important; color:#242628 !important;
    -webkit-text-fill-color:#242628 !important;
    font-family: Inter, sans-serif !important; font-size:14px !important;
    line-height:1.5 !important; padding-top: 7px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color:#8a8e93 !important; -webkit-text-fill-color:#8a8e93 !important; }
[data-testid="stChatInput"] button {
    width: 38px !important; height: 38px !important;
    background: var(--green) !important; color:white !important;
    border:0 !important; border-radius: 13px !important;
}
[data-testid="stChatInput"] button:hover { background: var(--green-2) !important; }

/* Anexos */
[data-testid="stChatInput"] [data-testid="stFileUploader"] { background: transparent !important; }

/* Scrollbar discreta */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #d7d9db; border-radius: 20px; }
::-webkit-scrollbar-thumb:hover { background: #c4c7c9; }

@media (max-width: 850px) {
    .brand-desc, .app-brand-tag { display:none; }
    .st-key-history_column, .st-key-current_column { padding: 0 14px; }
    .user-bubble { max-width: 86%; }
    [data-testid="stBottomBlockContainer"] > div { padding: 0 10px !important; }
    .st-key-welcome_wrap { padding: 24px 18px 10px; }
    .st-key-welcome_wrap [data-testid="stHorizontalBlock"] { flex-direction: column; }
}
</style>
""", unsafe_allow_html=True)
# ============================================================
# FUNÇÃO PUBMED
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
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
    # Coloque a sua chave do Pinecone aqui também (depois ensinarei a esconder isso por segurança)
    os.environ['PINECONE_API_KEY'] = 'pcsk_4wxRZ7_BmYg6AxjJCSSmBmjAqP7x5mipWZNQRiYH8PnHK6Eg2PosBBCMyXG46J7XzoyoA8'
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 🔴 APAGAMOS O CHROMA E AGORA LIGAMOS DIRETO NO PINECONE:
    banco_base = PineconeVectorStore(index_name="mentorvet", embedding=embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)
    
    return embeddings, banco_base, llm

embeddings, banco_base, llm = iniciar_cerebro()

@st.cache_resource
def iniciar_cadeia(llm):
    return llm | StrOutputParser()

cadeia = iniciar_cadeia(llm)


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
# INTERFACE — ESTILO CHATGPT / GEMINI
# ============================================================

# ------------------------------------------------------------
# Identidade visual no topo do painel principal
# (agora "sticky" e não mais sobreposta à sidebar)
# ------------------------------------------------------------
render_html(f"""
<div class="app-brand">
    <div class="app-brand-left">
        <div class="brand-mark">{STETHOSCOPE_SVG.format(size=16)}</div>
        <div>
            <div class="brand-name">MentorVet</div>
            <div class="brand-desc">Medicina Veterinária</div>
        </div>
    </div>
    <div class="app-brand-tag">IFMG Bambuí</div>
</div>
""")

# ------------------------------------------------------------
# Sidebar nativa do Streamlit, completamente redesenhada
# ------------------------------------------------------------
with st.sidebar:
    render_html(f"""
    <div class="sidebar-brand">
        <div class="sidebar-brand-mark">{STETHOSCOPE_SVG.format(size=18)}</div>
        <div>
            <div class="sidebar-brand-name">MentorVet</div>
            <div class="sidebar-brand-desc">Assistente acadêmico</div>
        </div>
    </div>
    """)

    if st.button("Nova conversa", use_container_width=True, key="new_chat", icon=":material/add:"):
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
            st.session_state.historico_conversas.insert(0, primeira_pergunta[:42] + ("…" if len(primeira_pergunta) > 42 else ""))
        st.session_state.mensagens = []
        st.session_state.pergunta_pendente = None
        if "banco_aluno" in st.session_state:
            del st.session_state.banco_aluno
        st.rerun()

    render_html('<div class="sidebar-label">Conversas recentes</div>')

    if st.session_state.historico_conversas:
        for i, conv in enumerate(st.session_state.historico_conversas[:8]):
            render_html(f'<div class="history-row"><span class="history-dot">•</span><span>{conv}</span></div>')
    else:
        render_html('<div class="sidebar-empty">Suas conversas aparecerão aqui.</div>')

    render_html('<div class="sidebar-spacer"></div>')

    if st.session_state.mensagens:
        df_historico = pd.DataFrame(st.session_state.mensagens)
        csv_data = df_historico.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Baixar conversa",
            data=csv_data,
            file_name="mentorvet_historico.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_chat",
            icon=":material/download:",
        )

    if st.button("Limpar conversa", use_container_width=True, key="clear_chat", icon=":material/delete:"):
        st.session_state.mensagens = []
        st.session_state.pergunta_pendente = None
        if "banco_aluno" in st.session_state:
            del st.session_state.banco_aluno
        st.rerun()

    render_html("""
    <div class="sidebar-footer">
        <div class="sidebar-footer-title">MentorVet</div>
        <div>Assistente acadêmico para Medicina Veterinária.</div>
    </div>
    """)

# ------------------------------------------------------------
# Conteúdo principal — tela inicial com sugestões reais
# (título + chips vivem no MESMO container, por isso ficam
# alinhados e centralizados como um bloco só)
# ------------------------------------------------------------
if not st.session_state.mensagens:
    with st.container(key="welcome_wrap"):
        render_html(f"""
        <div class="welcome-mark">{STETHOSCOPE_SVG.format(size=28)}</div>
        <h1>Como posso ajudar?</h1>
        <p>Estude Medicina Veterinária com apoio de uma base acadêmica e referências científicas.</p>
        """)

        sugestoes = [
            ("Cão, 6 anos, com poliúria, polidipsia e perda de peso. Diferenciais?", ":material/emergency:"),
            ("Mecanismo de ação da xilazina em pequenos animais.", ":material/medication:"),
            ("Como interpretar leucocitose com desvio à esquerda?", ":material/bloodtype:"),
            ("Protocolo de emergência para GDV em cães.", ":material/pets:"),
        ]

        col1, col2 = st.columns(2)
        for i, (texto, icone) in enumerate(sugestoes):
            coluna = col1 if i % 2 == 0 else col2
            with coluna:
                if st.button(texto, key=f"sugestao_{i}", use_container_width=True, icon=icone):
                    st.session_state.pergunta_pendente = texto
                    st.rerun()

# ------------------------------------------------------------
# Histórico — sem cartões pesados
# ------------------------------------------------------------
with st.container(key="history_column"):
    for msg in st.session_state.mensagens:
        if msg["role"] == "user":
            conteudo = msg["content"]
            texto = ""
            imagens = []
            if isinstance(conteudo, list):
                for item in conteudo:
                    if isinstance(item, str):
                        texto += item
                    elif hasattr(item, "save"):
                        imagens.append(item)
            else:
                texto = str(conteudo)

            render_html(f"""
            <div class="message-row user-row">
                <div class="user-bubble">{texto.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace(chr(10), '<br>')}</div>
            </div>
            """)
            for img in imagens:
                st.image(img, width=420)
        else:
            render_html(f"""
            <div class="message-row assistant-row">
                <div class="assistant-head">
                    <div class="assistant-mark">{STETHOSCOPE_SVG.format(size=15)}</div>
                    <span>MentorVet</span>
                </div>
            </div>
            """)
            st.markdown(msg["content"])

# ------------------------------------------------------------
# Input principal
# ------------------------------------------------------------
entrada_usuario = st.chat_input(
    "Pergunte ao MentorVet...",
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

# ------------------------------------------------------------
# Processamento — mesma lógica do projeto original
# ------------------------------------------------------------
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

    with st.container(key="current_column"):
        # Pergunta atual: bolha à direita, como chats modernos
        texto_visual = texto_pergunta.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace(chr(10), '<br>')
        render_html(f"""
        <div class="message-row user-row current-user-row">
            <div class="user-bubble">{texto_visual}</div>
        </div>
        """)
        if imagem_obj:
            st.image(imagem_obj, width=420)

        # Cabeçalho fixo da resposta
        render_html(f"""
        <div class="message-row assistant-row current-assistant-row">
            <div class="assistant-head">
                <div class="assistant-mark">{STETHOSCOPE_SVG.format(size=15)}</div>
                <span>MentorVet</span>
            </div>
        </div>
        """)

        # Indicador de carregamento em um placeholder de verdade,
        # para poder ser REMOVIDO assim que a resposta estiver pronta
        # (antes ele ficava preso na tela para sempre).
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            render_html("""
            <div class="loading-line">
                <span class="loading-dot"></span>
                <span class="loading-dot"></span>
                <span class="loading-dot"></span>
                <span>Preparando resposta</span>
            </div>
            """)

        contexto_textos = []
        try:
            docs_base = banco_base.similarity_search(texto_pergunta, k=8)
            contexto_textos.extend([d.page_content for d in docs_base])
        except Exception:
            pass

        if "banco_aluno" in st.session_state:
            try:
                docs_aluno = st.session_state.banco_aluno.similarity_search(texto_pergunta, k=6)
                contexto_textos.extend([d.page_content for d in docs_aluno])
            except Exception:
                pass

        texto_combinado = "\n\n".join(contexto_textos)

        if len(texto_combinado.strip()) < 150 and not imagem_obj:
            contexto_cientifico = buscar_pubmed(texto_pergunta)
            texto_combinado += f"\n\nLiteratura Científica Recente (PubMed):\n{contexto_cientifico}"

        prompt_especialista = f"""
            Você é o MentorVet, um Diplomado pelo Colégio Americano de Medicina Veterinária Interna (ACVIM) e preceptor sênior de excelência do IFMG Bambuí.
            Sua linguagem deve ser estritamente técnica, acadêmica e encorajadora. Nunca dê respostas rasas ou genéricas.

            Contexto extraído da biblioteca médica fixa (Use ISSO como sua verdade absoluta para doses e protocolos):
            {texto_combinado}

            Solicitação do Aluno / Descrição da Imagem:
            {texto_pergunta}

            DIRETRIZES DE RACIOCÍNIO CLÍNICO (Siga estritamente):
            1. Fundamentação Fisiopatológica: Antes de dar um diagnóstico ou tratamento, explique o "porquê". Detalhe a cascata fisiopatológica ou o mecanismo de ação dos fármacos envolvidos.
            2. Estrutura SOAP (Obrigatório para Casos Clínicos):
               - S (Subjetivo): Sintetize os achados da anamnese.
               - O (Objetivo): Interprete os exames de forma crítica (não apenas repita os valores, diga o que significam).
               - A (Avaliação): Liste diagnósticos diferenciais fundamentados, do mais provável ao menos provável.
               - P (Plano): Especifique terapias com doses exatas em mg/kg (busque no contexto fornecido), vias de administração e frequência.
            3. Resposta Direta (Perguntas teóricas): Se o aluno perguntar apenas uma dose ou conceito, responda diretamente com altíssimo rigor técnico, sem fazer o SOAP completo, mas cite o mecanismo de ação.
            4. Limites de IA: Se o caso exigir intervenção cirúrgica imediata ou referências que você não possui no contexto, alerte o aluno, mas ainda assim forneça a conduta de estabilização emergencial (Triage).
            """

        # Remove o indicador de carregamento antes de escrever a resposta final
        loading_placeholder.empty()

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
                resposta_final = st.write_stream(cadeia.stream(prompt_especialista))
            except Exception as erro:
                resposta_final = f"Não consegui gerar a resposta neste momento.\n\nErro: {erro}"
                st.error(resposta_final)

    st.session_state.mensagens.append({
        "role": "assistant",
        "content": resposta_final
    })