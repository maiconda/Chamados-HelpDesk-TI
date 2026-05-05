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

# Para salvar o modelo
import joblib
import re


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
    # plt.show()

def analise_contagens(df):
    # sns.barplot(x=df['Status'].value_counts().index, y=df['Status'].value_counts())
    # # plt.show()
    pass

# =========================
# PRÉ-PROCESSAMENTO
# =========================
def preprocessar_texto(df, stop_words):
    def limpar_texto(texto):
        texto = str(texto).lower()
        texto = re.sub(r'\d+', '', texto)
        texto = re.sub(r'[^\w\s]', '', texto)
        texto = " ".join([w for w in texto.split() if w not in stop_words])
        return texto

    df['Título_Limpo'] = df['Título'].apply(limpar_texto)
    return df

def analisar_categorias(df):
    print(df['Categoria'].value_counts())

def vetorizar(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Título_Limpo'])
    y = df['Categoria']
    return X, y, vectorizer

# =========================
# TREINAMENTO
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
        "Logistic Regression": lr.predict(X_test),
        "Naive Bayes": nb.predict(X_test),
        "Gradient Boosting": gb.predict(X_test.toarray())
    }

    for nome, y_pred in modelos.items():
        print(f"\n===== {nome} =====")
        print("Acurácia:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

# =========================
# PREVISÃO
# =========================
def prever(texto, vectorizer, modelos, stop_words):
    texto_limpo = texto.lower()
    texto_limpo = re.sub(r'\d+', '', texto_limpo)
    texto_limpo = re.sub(r'[^\w\s]', '', texto_limpo)
    texto_limpo = " ".join([w for w in texto_limpo.split() if w not in stop_words])

    vetor = vectorizer.transform([texto_limpo])

    pred_lr = modelos["Logistic Regression"].predict(vetor)[0]
    pred_nb = modelos["Naive Bayes"].predict(vetor)[0]
    pred_gb = modelos["Gradient Boosting"].predict(vetor.toarray())[0]

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
    url = "https://gist.githubusercontent.com/felipebfava/9b465894e64a6bb4f26a4e144c23adf6/raw/6ef001f1f4b6022c74a15556295f1f9cc2b8d884/gistfile1.txt"

    stop_words = carregar_stopwords()

    df = carregar_dados(url)
    df = anonimizar_dados(df)

    # analise_exploratoria(df) # Removido # plt.show() para execução automática

    # analise_contagens(df)

    df = preprocessar_texto(df, stop_words)

    # analisar_categorias(df)

    X, y, vectorizer = vetorizar(df)

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

    # =========================
    # SALVAR MODELO E VETORIZADOR (ETAPA 1)
    # =========================
    print("\nSalvando modelo e vetorizador...")
    joblib.dump(lr, 'modelo_regressao_logistica.pkl')
    joblib.dump(vectorizer, 'vetorizador_tfidf.pkl')
    print("Arquivos 'modelo_regressao_logistica.pkl' e 'vetorizador_tfidf.pkl' salvos com sucesso!")

if __name__ == "__main__":
    main()
