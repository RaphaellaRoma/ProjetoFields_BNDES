import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, precision_recall_curve
import joblib 
import os 

## Preparando o DataFrame 
df = pd.DataFrame(pd.read_csv('./Dados CSV/df_assunto_limpo.csv'))
X = df["assunto_normativo"].astype(str)
y = df["aplicavel_bndes"].map({'Sim': 1, 'Não': 0}) 

# Garantindo que X e y tenham o mesmo tamanho
y = y.dropna() 
X = X[y.index]

## Separando em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y 
)


## Vetorizando os textos 
# Tokeniza o texto 
vectorizer = TfidfVectorizer(ngram_range=(1, 2)) # Inclui palavras únicas e pares de palavras 
# Calcula a importância de cada palavra/token (IDFs) e cria uma matriz 
X_train_tfidf = vectorizer.fit_transform(X_train)
# Usa a importância aprendida para criar a matriz 
X_test_tfidf = vectorizer.transform(X_test)


## Treinamento e teste com a Rede Neural 
model = MLPClassifier(
    hidden_layer_sizes=(50, 30),  
    activation="relu",
    solver="adam",
    max_iter=500, 
    random_state=42,
    early_stopping=True,
    alpha=0.01
)
model.fit(X_train_tfidf, y_train)
y_proba = model.predict_proba(X_test_tfidf)[:, 1]


## Analisando resultados 
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Melhor threshold pelo F1
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = f1_scores.argmax()
if best_idx < len(thresholds):
    best_threshold = thresholds[best_idx]
else:
    best_threshold = thresholds[-1] 
print(f"Melhor threshold pelo F1: {best_threshold:.2f}")

# Ajuste manual de threshold
y_pred_adj = (y_proba >= best_threshold).astype(int)
print("Threshold ajustado:")
print(classification_report(y_test, y_pred_adj))


## Salvando os artefatos 
SAVE_DIR_NN = "./modelo_nn_salvo"
os.makedirs(SAVE_DIR_NN, exist_ok=True)
print(f"\nSalvando artefatos em {SAVE_DIR_NN}...")

# 1. Salvar o vetorizador
joblib.dump(vectorizer, f"{SAVE_DIR_NN}/vectorizer.pkl")
print("Vetorizador salvo.")

# 2. Salvar o modelo
joblib.dump(model, f"{SAVE_DIR_NN}/model.pkl")
print("Modelo MLP salvo.")

# 3. Salvar o threshold
joblib.dump(best_threshold, f"{SAVE_DIR_NN}/threshold.pkl")
print(f"Threshold {best_threshold:.4f} salvo.")

