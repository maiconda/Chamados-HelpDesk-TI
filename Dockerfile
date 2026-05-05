# Use uma imagem base oficial do Python
FROM python:3.12-slim

# Define o diretório de trabalho no container
WORKDIR /app

# Copia o arquivo de dependências para o container
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Faz o download das stopwords do NLTK durante o build
RUN python -c "import nltk; nltk.download('stopwords')"

# Copia os arquivos do projeto para o diretório de trabalho
COPY app.py .
COPY modelo_regressao_logistica.pkl .
COPY vetorizador_tfidf.pkl .

# Expõe a porta que o Streamlit usa por padrão
EXPOSE 8501

# Configurações para otimizar o Streamlit em produção e permitir Proxy (Cloudflare)
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Comando para rodar a aplicação
CMD ["streamlit", "run", "app.py"]
