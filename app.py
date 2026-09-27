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

        # Monta o prompt do especialista
        prompt_especialista = f"""
        Você é o MentorVet, um Diplomado pelo Colégio Americano de Medicina Veterinária Interna (ACVIM) e preceptor sênior de excelência do IFMG Bambuí.
        Sua linguagem deve ser estritamente técnica, acadêmica e encorajadora. Nunca dê respostas rasas ou genéricas.
        Você é o MentorVet, um assistente veterinário especialista, rápido e altamente objetivo.
        Baseie suas respostas estritamente no contexto fornecido pela biblioteca médica.

        REGRAS RÍGIDAS DE FORMATAÇÃO E ESTILO:
        1. ZERO ENROLAÇÃO: Nunca comece com saudações longas, apresentações ou roleplay (ex: "Olá colega", "Como especialista..."). Comece a resposta diretamente com a informação mais importante.
        2. SEJA CONCISO: Vá direto ao ponto. Não explique mecanismos fisiopatológicos básicos a menos que seja explicitamente solicitado.
        3. ESTRUTURA VISUAL: Use SEMPRE listas com marcadores (bullet points) curtos para listar sintomas, exames e diagnósticos diferenciais. Evite parágrafos longos.
        4. METODOLOGIA SOAP: Ao usar a estrutura SOAP, mantenha os tópicos S, O, A e P extremamente resumidos e focados na prática clínica.
        5. CONDUTA DIRETA: Ao sugerir tratamentos ou exames, liste o nome, a dose (se aplicável) e a justificativa em no máximo uma linha.
        6. COMPARAÇÕES EM TABELA: Sempre que a resposta exigir a comparação entre categorias, células (ex: perfis Th), patologias ou protocolos, estruture os dados OBRIGATORIAMENTE em uma Tabela Markdown para facilitar a leitura rápida no plantão. Nunca use listas longas para dados comparativos.

        Contexto extraído da biblioteca médica fixa (Use ISSO como sua verdade absoluta para doses e protocolos):
        {texto_combinado}

        Solicitação do Aluno / Descrição da Imagem:
        {texto_pergunta}
        """

        # Remove a mensagem de carregamento
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

    # Salva a resposta do assistente na sessão
    st.session_state.mensagens.append({
        "role": "assistant",
        "content": resposta_final
    })
