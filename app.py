import streamlit as st
import joblib
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# ==========================================
# CONFIGURAÇÕES DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="HelpDesk TI - Classificador",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# FUNÇÕES DE APOIO
# ==========================================

@st.cache_resource
def carregar_recursos():
    """Carrega o modelo e o vetorizador uma única vez."""
    modelo = joblib.load('modelo_regressao_logistica.pkl')
    vectorizer = joblib.load('vetorizador_tfidf.pkl')
    
    try:
        stop_words = set(stopwords.words('portuguese'))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('portuguese'))
        
    return modelo, vectorizer, stop_words

def limpar_texto(texto, stop_words):
    """Mesma lógica de limpeza usada no treinamento."""
    texto = str(texto).lower()
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = " ".join([w for w in texto.split() if w not in stop_words])
    return texto

# ==========================================
# INTERFACE PRINCIPAL
# ==========================================

def main():
    modelo, vectorizer, stop_words = carregar_recursos()

    # --- Sidebar ---
    st.sidebar.title("Sobre o Projeto")
    st.sidebar.info("""
    **Classificador de Chamados TI**
    
    Este modelo utiliza Processamento de Linguagem Natural (PLN) para prever a categoria de um chamado com base no seu título.
    
    **Integrantes:**
    - Felipe Biava Favarin
    - Maicon de Oliveira da Silva
    - Gabriel Nogueira
    """)
    
    st.sidebar.metric("Acurácia do Modelo", "67.5%")

    # --- Cabeçalho ---
    st.title("🤖 Classificador Automático de Chamados")
    st.markdown("""
    Esta ferramenta ajuda a triagem do HelpDesk, sugerindo a categoria mais provável para um novo problema relatado.
    """)

    # --- Área de Input ---
    st.subheader("Novo Chamado")
    titulo_input = st.text_input(
        "Digite o título ou descrição curta do problema:",
        placeholder="Ex: Minha internet parou de funcionar no bloco B"
    )

    if st.button("Classificar Categoria"):
        if titulo_input:
            # Processamento
            texto_processado = limpar_texto(titulo_input, stop_words)
            vetor = vectorizer.transform([texto_processado])
            
            # Previsão
            previsao = modelo.predict(vetor)[0]
            probabilidades = modelo.predict_proba(vetor)
            confianca = max(probabilidades[0]) * 100

            # Exibição do Resultado
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"### Categoria Prevista: **{previsao}**")
            
            with col2:
                st.metric("Nível de Confiança", f"{confianca:.2f}%")
                
            st.info(f"**Texto processado pelo modelo:** {texto_processado}")
        else:
            st.warning("Por favor, digite um título para realizar a classificação.")

    # --- Rodapé ---
    st.markdown("---")
    st.caption("Desenvolvido para a disciplina de Inteligência Artificial / Ciência de Dados.")

if __name__ == "__main__":
    main()
