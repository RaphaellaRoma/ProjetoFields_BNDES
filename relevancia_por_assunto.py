import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, precision_recall_curve


## Preparando o DataFrame 

df = pd.DataFrame(pd.read_excel('df_atualizado.xlsx'))
# Linhas sem assunto não são usadas 
df = df[df["Assunto do Normativo"] != "..."]
# Normas não aplicaveis não são usadas 
df = df[df['1ª Avaliação de Relevância (AIC - Time de Compliance)'].astype(str) != "0"]

X = df["Assunto do Normativo"].astype(str)
y = df["1ª Avaliação de Relevância (AIC - Time de Compliance)"].map({'Baixa': 1, 'Média': 2, 'Alta': 3})

# Garantindo que X e y tenham o mesmo tamanho
y = y.dropna() 
X = X[y.index]


## Separando em Treino e Teste

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y 
)


## Vetorizando os textos 

# Tokeniza o texto 
vectorizer = TfidfVectorizer()
# Calcula a importância de cada palavra/token (IDFs) e cria uma matriz 
X_train_tfidf = vectorizer.fit_transform(X_train)
# Usa a importância aprendida para criar a matriz 
X_test_tfidf = vectorizer.transform(X_test)


## Treinamento e teste com a Rede Neural 

model = MLPClassifier(
    hidden_layer_sizes=(150, 100),
    activation="relu",
    solver="adam",
    max_iter=500, 
    random_state=42,
    early_stopping=True,
    alpha=0.001
)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)


## Analisando resultados 

print(classification_report(
    y_test, 
    y_pred, 
    target_names=['Baixa (1)', 'Média (2)', 'Alta (3)']
))
