Integrantes do Grupo:

- FELIPE BIAVA FAVARIN
- MAICON DE OLIVEIRA DA SILVA
- GABRIEL NOGUEIRA

In[ ]:

Para criar um arquivo de texto com os requisitos / bibliotecas para usar o ambiente Jupyter Notebook
Após esse comando, no terminal do VsCode, rode: pip install -r requirements.txt
get_ipython().system('pip freeze > requirements.txt')

Escolha do Problema:

- Classificação de Categoria: Prever a categoria do chamado com base no título

### Preparação e Tratamento dos Dados

A tabela foi importada como .csv e tratada via LibreOffice Calc.

Foi ajustado a coluna ID, que será para dados do tipo Número, estavam dispostos da maneira "10 552" e não como "10.552" ou "10552". Sendo necessário o ajuste, para a formatação entender que se tratava de um número e não um texto. Assim, também houve ajuste nas demais colunas.

Foram separadas as datas e horas das colunas "Data de abertura" e "Última atualização" para melhor tratamento e para quaisquer análises gráficas.

Eliminamos a coluna "Unnamed: 10", pois não apresentou nenhum dado relevante para o estudo.

Acrescentamos a coluna "Duração do Chamado em dias" para sabermos a quantidade de dias em que o chamado ficou aberto até o momento da última atualização.

Nas colunas 'Requerente - Requerente' e 'Atribuído - Técnico' geramos nomes falsos para compor a coluna. Utilizando a biblioteca Faker do Python.

Ajustamos a coluna Prioridade, atribuindo um valor numérico ao invés do texto, ficando organizada da seguinte forma:

- Muito baixa: 0
- Baixa: 1
- Média: 2
- Alta: 3
- Muito alta: 4
- Crítica: 5

Posteriormente, somente as colunas Título e Categoria ficam no dataset, todas as outras são excluídas / dropadas. Isto acontece para o atendimento do problema proposto:

- Classificação de Categoria: Prever a categoria do chamado com base no título

### Tratamento da Coluna Categoria

Foi observado valores nulos e valores de categorias que só apareciam uma única vez, tendo baixa incidência, isso afetará nosso algoritmo modelo escolhido na hora do treinamento / teste. Porém, foi mantido categorias com até 3 ocorrências, como a Categoria 'AUDIOVISUAL+TI'.

Então, nas linhas onde as categorias são:

- NaN
- EVENTOS+AUDIOVISUAL
- TI > Cabeamento Estruturado
- TI > Sistemas > Moodle
- EVENTOS
- LABORATÓRIOS DE ELETRO

Foram retiradas do dataset.

Além disso, algumas categorias com poucas ocorrências eram divisões de outras categorias maiores. Assim, optamos por fazer as seguintes substituições:

- TI > Rede > Melhorias performance > Segurança --> TI
- TI > Rede > Melhorias performance --> TI
- TI > Servidores --> TI

Isso irá nos ajudar nos treinamentos e testes dos modelos.

Para tratar a coluna 'Título' usaremos as técnicas de pré-processamento de texto e vetorização TF-IDF. Tratando os dados do texto para posteriormente serem treinados.

### Instalação das Bibliotecas
Execute no terminal:

```py
pip install pandas numpy matplotlib seaborn faker nltk scikit-learn
```

#### Seleção, Limpeza e Pré-processamento de Texto

Realização das seguintes etapas:
1. Seleciona somente as Categorias importantes para análise
2. Faz a Limpeza de Texto:
 - Converte o texto para minúsculas.
 - Remove caracteres não alfabéticos (mantendo apenas letras e espaços).
 - Tokeniza o texto em palavras.
 - Remove stopwords (palavras comuns que não adicionam / alteram significado, como 'de', 'a', 'o').
 - Junção das palavras processadas em uma string.

Análise da quantidade de chamados por Prioridade
A coluna Prioridade está organizada da seguinte forma:
Muito baixa: 0
Baixa: 1
Média: 2
Alta: 3
Muito alta: 4
Crítica: 5

## Resumo

### O que aconteceu no geral?

O principal problema:

 - Desbalanceamento extremo das classes + poucos dados

Impacto direto:

- Os modelos aprenderam bem em classes grandes como INFRA e TI
- Ignoraram classes pequenas com poucos exemplos

Isso gerou:
- precisão = 0 nas classes
- warnings (UndefinedMetricWarning)

Ou seja:

- Os modelos simplesmente não aprenderam nada sobre algumas categorias


### Qual modelo apresentou melhores resultados?
#### Regressão Logística foi o Melhor modelo

- Acurácia: 0.67 (maior)

- Melhor equilíbrio geral

- Conseguiu prever mais classes diferentes

Destaque:

- INFRA → muito bom (0.79 f1)
- TI → razoável (0.66 f1)
- INFRA+TI → bom (0.67 f1)

Problema:

- Ignorou classes pequenas (0.00)

Conclusão: modelo mais confiável do experimento

#### Gradient Boosting
- Acurácia: 0.57

- Desempenho mediano

Destaque:

- INFRA → bom (0.70 f1)
- LABORATÓRIOS → razoável (0.62 f1)

Problema:

- Baixa generalização
- Muitas classes com 0.00

Conclusão: não ideal para texto

#### Naive Bayes Multinomial
- Acurácia: 0.56

- Forte viés para classe majoritária

Comportamento claro:

- INFRA → recall = 1.00 (acertou quase tudo)

Mas:
- ignorou quase todas as outras classes

Isso significa:

- Ele está classificando quase tudo como INFRA

Conclusão: modelo simples, mas enviesado

### Sobre as métricas de avaliação dos modelos:
#### Accuracy (Acurácia)
É a proporção de todas as classificações que estavam corretas, sejam elas positivas ou negativas. Mostra o Total de acertos pelo total de exemplos

No nosso caso a melhor acurácia foi de 67%, isto é:

- 67% dos chamados foram classificados corretamente

Um modelo perfeito teria zero falso positivo e negativo e, portanto, uma acurácia de 1,0 ou 100%.

No entanto, quando o conjunto de dados está desequilibrado ou quando um tipo de erro (FN ou FP) é maior que o outro, é melhor otimizar uma das outras métricas.

#### Precision (Precisão)
Possui foco nos dados que podem ser falsos positivos.

É a proporção de todas as classificações positivas do modelo que são realmente positivas.

Um modelo perfeito hipotético teria zero falso positivo e, portanto, uma precisão de 1,0.

Em um conjunto de dados desequilibrado em que o número de positivos reais é muito baixo, 1 a 2 exemplos no total, a precisão é menos significativa e menos útil.

A precisão melhora à medida que os falsos positivos diminuem, enquanto o recall melhora quando os falsos negativos diminuem.

Então:
- Precisão baixa = muitos falsos positivos

#### Recall
Recall ou também conhecida como taxa de verdadeiro positivo (TVP), ou a proporção de todos os positivos reais que foram classificados corretamente como positivos.

Um modelo perfeito hipotético teria zero falsos negativos e, portanto, um recall (TPR) de 1,0, ou seja, uma taxa de detecção de 100%.

Então:
- Recall baixo = modelo está “perdendo” exemplos

Em um conjunto de dados desbalanceado em que o número de positivos reais é muito baixo, o recall é uma métrica mais significativa do que a acurácia, porque mede a capacidade do modelo de identificar corretamente todas as instâncias positivas.

#### F1-score
Faz uma média entre a Precisão e Recall. Varia de 0 a 1, onde 1 indica desempenho perfeito e 0 indica um desempenho ruim.

Melhor métrica para comparar modelos

É especialmente útil em cenários com dados desbalanceados, onde uma classe tem muito mais exemplos que a outra.

#### Support (Suporte)
Nos mostra o número de ocorrências reais de cada classe no conjunto de dados de teste. Uma quantidade real de exemplos de cada classe.

O suporte não é uma métrica de desempenho (como a acurácia), mas sim uma métrica de contexto que indica a distribuição real dos dados. Isto é, mostra o desbalanceamento deles.

### Por que esses 3 modelos foram escolhidos?
#### Logistic Regression
- Padrão em PLN
- Funciona muito bem com TF-IDF
- Linear → bom para dados esparsos

#### Naive Bayes (Multinomial)
- Clássico para texto
- Baseado em probabilidade
- Muito rápido e eficiente

#### Gradient Boosting
- Modelo mais avançado (ensemble)
- Incluído para comparação
- Mostra diferença entre: modelos lineares vs não-lineares

### Quais outros modelos poderiam ser usados?
#### SVM (LinearSVC)
- Geralmente um dos melhores para texto
- Muito usado em classificação de texto

#### Random Forest
- Alternativa ao Gradient Boosting
- Melhor para dados tabulares

#### Redes neurais (MLP)
- Podem capturar padrões mais complexos
- Mas precisam de mais dados

#### Modelos modernos (extra)
- BERT (transformers)
- Word embeddings