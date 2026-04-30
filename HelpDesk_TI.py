
# =========================
# INTEGRANTES DO GRUPO
# =========================
# - FELIPE BIAVA FAVARIN
# - MAICON DE OLIVEIRA DA SILVA
# - GABRIEL NOGUEIRA

# =========================
# IMPORTAÇÃO DAS BIBLIOTECAS
# =========================
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Para gerar dados falsos
from faker import Faker

# Para divisão treino/teste e métricas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Modelos de avaliação escolhidos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

# Para PLN do texto e vetorização
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# CONFIGURAÇÃO PARA PLN
# =========================
def carregar_stopwords():
    try:
        return set(stopwords.words('portuguese'))
    except:
        nltk.download('stopwords')
        return set(stopwords.words('portuguese'))

# =========================
# CARREGAMENTO
# =========================
def carregar_dados(url):
    # Visualização da Tabela carregada via URL
    df = pd.read_csv(url)
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    return df

# =========================
# ANONIMIZAÇÃO DOS DADOS
# =========================

# O Faker está atribuindo nomes aleatórios nas colunas 'Requerente - Requerente' e 'Atribuído - Técnico'.
def anonimizar_dados(df):
    # Inicializa o Faker
    fake = Faker('pt_BR')

    # Preenche valores nulos com string vazia para as colunas relevantes
    df['Requerente - Requerente'] = df['Requerente - Requerente'].fillna('')
    df['Atribuído - Técnico'] = df['Atribuído - Técnico'].fillna('')

    # Mapeamento para Requerentes (de identificador para nome falso)
    requerente_name_mapping = {}
    unique_requerentes_ids = df['Requerente - Requerente'].unique()
    for req_id in unique_requerentes_ids:
        if req_id == '':
            requerente_name_mapping[req_id] = ''
        else:
            requerente_name_mapping[req_id] = fake.name()

    print(df[['Requerente - Requerente', 'Atribuído - Técnico']].head())
    return df


# =========================
# ANÁLISE EXPLORATÓRIA
# =========================
def analise_exploratoria(df):
    media_duracao_chamado = df['Duração do Chamado em dias'].mean()
    print(f"Média da Duração do Chamado em dias: {media_duracao_chamado:.2f} dias")

    ## Histograma
    # Análise da quantidade de dias que um chamado está registrado no sistema

    # Necessário criar um DataFrame sem a coluna 'ID' para o histograma. Assim obtemos somente a variável numérica 'Duração do Chamado em dias'.
    df_sem_id = df.drop(columns=['ID'], errors='ignore')

    df_sem_id.hist(figsize=(15,10)) # largura / altura
    plt.show()


    # sns.barplot(x=df['Status'].value_counts().index,
    #             y=df['Status'].value_counts().values)
    # plt.show()

    # Gráfico de Barras
    # Análise de como os chamados estão distribuídos e quais suas classes, entre 'Em atendimento (atribuído)', 'Pendente' e 'Em atendimento (planejado)'
    qtd_status = df['Status'].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=qtd_status.index, y=qtd_status.values, hue=qtd_status.index, palette='muted', legend=False)
    plt.title('Distribuição dos Status dos Chamados')
    plt.xlabel('Status')
    plt.ylabel('Número de Chamados')
    plt.tight_layout()
    plt.show()

# =========================
# CONTAGENS IMPORTANTES
# =========================
def analise_contagens(df):
    # Análise da quantidade de Requerentes e de Técnicos presentes
    num_requerentes = df['Requerente - Requerente'].nunique()
    num_tecnicos = df['Atribuído - Técnico'].nunique()

    print(f"Quantidade de Requerentes diferentes: {num_requerentes}")
    print(f"Quantidade de Técnicos diferentes: {num_tecnicos}")

    # Análise da quantidade de chamados feitos por cada Requerente
    print("\nQuantidade de chamados por Requerente:")
    print(df['Requerente - Requerente'].value_counts())

    # Análise da quantidade de chamados atribuídos a cada Técnico
    print("\nQuantidade de chamados por Técnico:")
    print(df['Atribuído - Técnico'].value_counts())

    # Análise da quantidade de chamados por Prioridade
    # A coluna Prioridade está organizada da seguinte forma:
    # Muito baixa: 0, Baixa: 1, Média: 2, Alta: 3, Muito alta: 4, Crítica: 5

    print("\nQuantidade de chamados por Prioridade:")
    print(df['Prioridade'].value_counts().sort_index()) # sort_index organiza em ordem crescente

    # Para verificar em uma Prioridade específica
    outra_prioridade = df[df['Prioridade'] == 2].shape[0]
    print(f"\nQuantidade de chamados com Prioridade '2': {outra_prioridade}")

# =========================
# PRÉ-PROCESSAMENTO
# =========================
def preprocessar_texto(df, stop_words):
    ##Seleção
    df = df[['Título', 'Categoria']]
    # remove valores nulos
    df = df.dropna()

    print("\nQuantidade de dados:", len(df))
    print("\n")

    # transforma o texto em minúsculas
    df['Titulo_limpo'] = df['Título'].str.lower()
    
    # remove números
    df['Titulo_limpo'] = df['Titulo_limpo'].str.replace(r'\d+', '', regex=True)
    
    # remove pontuação
    df['Titulo_limpo'] = df['Titulo_limpo'].str.replace(r'[^\w\s]', '', regex=True)

    # remove stopwords
    df['Titulo_limpo'] = df['Titulo_limpo'].apply(
        lambda t: " ".join([w for w in t.split() if w not in stop_words])
    )

    print(df.head())
    return df

# =========================
# DISTRIBUIÇÃO DE CATEGORIAS
# =========================
def analisar_categorias(df):
    print("\nDistribuição das categorias:")
    print(df['Categoria'].value_counts())


# =========================
# VETORIZAÇÃO
# =========================
def vetorizar(df):
    ## Vetorização (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['Titulo_limpo'])
    y = df['Categoria']

    # Shape nos mostra a tupla (linhas, colunas)
    # Linhas são o número de títulos de chamados processados.
    # Colunas são o número de características (features) únicas que foram extraídas do texto.
    print("Shape:", X.shape)
    return X, y, vectorizer

# =========================
# TREINO
# =========================
def treinar_modelos(X_train, y_train):
    # Regressão Logística
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)

    # Naive Bayes Multinomial
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    
    # Gradient Boosting
    gb = GradientBoostingClassifier()
    gb.fit(X_train.toarray(), y_train)

    return lr, nb, gb

# =========================
# AVALIAÇÃO
# =========================
def avaliar_modelos(lr, nb, gb, X_test, y_test):
    modelos = {
        "Logistic Regression": lr.predict(X_test),              # Avaliação: Regressão Logística
        "Naive Bayes": nb.predict(X_test),                      # Avaliação: Naive Bayes Multinomial
        "Gradient Boosting": gb.predict(X_test.toarray())       # Avaliação: Gradient Boosting
    }

    for nome, y_pred in modelos.items():
        print(f"\n===== {nome} =====")
        print("Acurácia:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

# =========================
# PREVISÃO
# =========================
def prever(texto, vectorizer, modelos, stop_words):
    # Aplica mesma limpeza de texto usada no treino do dataset
    texto_limpo = texto.lower()
    texto_limpo = re.sub(r'\d+', '', texto_limpo)
    texto_limpo = re.sub(r'[^\w\s]', '', texto_limpo)
    texto_limpo = " ".join([w for w in texto_limpo.split() if w not in stop_words])

    # Vetorização
    vetor = vectorizer.transform([texto_limpo])

    # Previsões
    print("\nTexto:", texto)
    for nome, modelo in modelos.items():
        if nome == "Gradient Boosting":
            pred = modelo.predict(vetor.toarray())[0]
        else:
            pred = modelo.predict(vetor)[0]

        print(nome, ":", pred)


    pred_lr = modelos["Logistic Regression"].predict(vetor)[0]
    pred_nb = modelos["Naive Bayes"].predict(vetor)[0]
    pred_gb = modelos["Gradient Boosting"].predict(vetor.toarray())[0]

    # Resultado
    print("\n============================")
    print("Texto original:", texto)
    print("Texto limpo:", texto_limpo)
    print("\nPrevisões:")
    print("Logistic Regression:", pred_lr)
    print("Naive Bayes:", pred_nb)
    print("Gradient Boosting:", pred_gb)

# =========================
# MAIN
# =========================
def main():
    # Definindo a URL do dataset (link raw atualizado) baseado no Github Gists
    # https://gist.github.com/felipebfava/9b465894e64a6bb4f26a4e144c23adf6
    url = "https://gist.githubusercontent.com/felipebfava/9b465894e64a6bb4f26a4e144c23adf6/raw/6ef001f1f4b6022c74a15556295f1f9cc2b8d884/gistfile1.txt"

    stop_words = carregar_stopwords()

    df = carregar_dados(url)
    df = anonimizar_dados(df)

    analise_exploratoria(df)

    analise_contagens(df)

    df = preprocessar_texto(df, stop_words)

    analisar_categorias(df)

    X, y, vectorizer = vetorizar(df)

    ## Divisão treino/teste numa proporção de 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    lr, nb, gb = treinar_modelos(X_train, y_train)

    avaliar_modelos(lr, nb, gb, X_test, y_test)

    modelos = {
        "Logistic Regression": lr,
        "Naive Bayes": nb,
        "Gradient Boosting": gb
    }

    prever("computador não liga no laboratório", vectorizer, modelos, stop_words)
    prever("projetor não está funcionando na sala", vectorizer, modelos, stop_words)
    prever("internet caiu no bloco administrativo", vectorizer, modelos, stop_words)

if __name__ == "__main__":
    main()
