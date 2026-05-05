# Documentação de Implementação: Interface Web HelpDesk TI

Este documento registra o passo a passo da evolução do projeto, desde o script de análise original até a implementação da interface gráfica.

---

## Etapa 1: Persistência do Modelo (Model Persistence)

**Objetivo:** Exportar o modelo treinado e o vetorizador para arquivos físicos, permitindo que a interface gráfica carregue o "cérebro" do projeto instantaneamente sem necessidade de re-treinamento.

### 1. Preparação do Ambiente
Foram instaladas as bibliotecas necessárias para a persistência e para a futura interface:
- **joblib:** Utilizada para salvar e carregar objetos Python (modelos e transformadores).
- **streamlit:** Biblioteca principal para a criação da interface web.

**Comando utilizado:**
```bash
pip install joblib streamlit
```

### 2. Modificações no Código (`HelpDesk_TI.py`)
O script original foi atualizado para incluir a lógica de exportação. As principais mudanças foram:
- Inclusão do `import joblib`.
- Adição de um bloco de código ao final da função `main()` para salvar o modelo de **Regressão Logística** (identificado como o de melhor performance nas métricas) e o **TfidfVectorizer**.

**Trecho de código adicionado:**
```python
# SALVAR MODELO E VETORIZADOR
print("\nSalvando modelo e vetorizador...")
joblib.dump(lr, 'modelo_regressao_logistica.pkl')
joblib.dump(vectorizer, 'vetorizador_tfidf.pkl')
print("Arquivos 'modelo_regressao_logistica.pkl' e 'vetorizador_tfidf.pkl' salvos com sucesso!")
```

### 3. Execução e Validação
O script foi executado via terminal para processar os dados, treinar os modelos e gerar os arquivos.

**Comando de execução:**
```bash
python3 HelpDesk_TI.py
```

**Resultado da Validação:**
- A execução confirmou a acurácia de **~67%** para a Regressão Logística.
- Os arquivos `modelo_regressao_logistica.pkl` e `vetorizador_tfidf.pkl` foram criados com sucesso no diretório raiz do projeto.
- O modelo persistido foi testado com a frase *"computador não liga no laboratório"*, retornando a categoria correta: `LABORATÓRIOS DE INFORMÁTICA`.

---
*Próxima etapa: Desenvolvimento da Interface com Streamlit.*

## Etapa 2: Desenvolvimento da Interface (Streamlit App)

**Objetivo:** Criar uma aplicação web interativa que permita aos usuários finais utilizar o modelo treinado de forma simples, sem necessidade de conhecimento técnico ou execução de scripts via terminal.

### 1. Criação do Aplicativo (`app.py`)
Foi desenvolvido um novo arquivo Python focado exclusivamente na interface do usuário. As principais funcionalidades implementadas foram:
- **Cache de Recursos:** Utilização do `@st.cache_resource` para carregar o modelo (`.pkl`) e o vetorizador apenas uma vez, otimizando a memória e a velocidade de resposta.
- **Lógica de Limpeza Espelhada:** Replicação exata da função `limpar_texto` utilizada no treinamento para garantir a consistência das previsões.
- **Métricas de Confiança:** Além da categoria prevista, o app calcula e exibe a probabilidade (confiança) da previsão através do método `predict_proba`.

### 2. Elementos Visuais
- **Barra Lateral:** Exibe informações sobre o projeto, integrantes do grupo e a acurácia global do modelo (67.5%).
- **Campo de Entrada:** `st.text_input` amigável com placeholders de exemplo.
- **Feedback Visual:** Uso de `st.success` para o resultado positivo e `st.metric` para o nível de confiança.

### 3. Execução e Validação
A interface foi testada localmente para garantir que a comunicação com os arquivos persistidos na Etapa 1 estava funcional.

**Comando de execução:**
```bash
streamlit run app.py
```

**Resultado da Validação:**
- O sistema carregou instantaneamente.
- Testes com entradas reais (ex: "Internet lenta no bloco A") retornaram previsões coerentes com o dataset original.
- O nível de confiança forneceu um feedback valioso sobre a certeza do modelo em cada classificação.

---
*Status do Projeto: Implementação Concluída.*

## Etapa 3: Containerização (Docker)

**Objetivo:** Empacotar a aplicação em um container Docker, garantindo que ela funcione em qualquer ambiente de produção com todas as dependências isoladas.

### 1. Arquivos Criados
- **`requirements.txt`:** Lista as bibliotecas Python necessárias (`streamlit`, `joblib`, `pandas`, `scikit-learn`, `nltk`).
- **`.dockerignore`:** Garante que arquivos de desenvolvimento (como `venv/`, notebooks e scripts de treino) não sejam enviados para a imagem final, mantendo-a leve.
- **`Dockerfile`:** Instruções para construir a imagem baseada em `python:3.12-slim`, instalando dependências e configurando o servidor Streamlit para modo produção.

### 2. Comandos para Build e Execução

**Para construir a imagem:**
```bash
docker build -t helpdesk-ti-app .
```

**Para rodar o container:**
```bash
docker run -p 8501:8501 helpdesk-ti-app
```
A aplicação ficará disponível em `http://localhost:8501`.

### 3. Vantagens
- **Portabilidade:** Funciona em qualquer servidor que suporte Docker (AWS, Azure, Google Cloud, etc.).
- **Isolamento:** Evita conflitos de versões de bibliotecas no servidor.
- **Pronto para Produção:** Configurado em modo *headless* para melhor performance.

---

## Etapa 4: Orquestração com Docker Compose

**Objetivo:** Facilitar o processo de build e execução da aplicação através de um único arquivo de configuração, além de incluir ajustes necessários para funcionamento atrás de túneis (Cloudflare).

### 1. Ajustes de Proxy (Cloudflare Tunnel)
O `Dockerfile` foi atualizado com as seguintes variáveis de ambiente para permitir a comunicação via WebSockets e evitar bloqueios de segurança do Streamlit ao usar domínios externos:
- `STREAMLIT_SERVER_ENABLE_CORS=false`
- `STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false`

### 2. Docker Compose (`docker-compose.yml`)
Criamos um arquivo de orquestração que automatiza o mapeamento de portas e reinicialização automática do container.

**Comandos Simplificados:**

- **Para construir e subir a aplicação:**
```bash
docker-compose up -d --build
```

- **Para parar a aplicação:**
```bash
docker-compose down
```

### 3. Benefícios
- **Automação:** Não é mais necessário digitar comandos longos com flags `-p` ou `-e`.
- **Persistência:** O container está configurado para reiniciar sozinho caso o servidor reinicie (`restart: always`).
