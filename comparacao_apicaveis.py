import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier


df = pd.read_csv('normativos_processados.csv')

X = df["texto_normativo"].astype(str)
y = df["aplicavel_bndes"].astype(int)  # 0/1


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Vetorização TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)



modelos = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(10, 5), activation="relu",
                         solver="adam", max_iter=300, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost" : XGBClassifier(n_estimators=300, learning_rate=0.1,max_depth=6, random_state=42,
                        use_label_encoder=False, eval_metric="logloss")
}



comparativo = []

melhor_f1 = -1
melhor_modelo_nome = ""
melhor_modelo_pred = None
melhor_threshold = 0

for nome, modelo in modelos.items():
    modelo.fit(X_train_tfidf, y_train)
    
    # Probabilidades
    if hasattr(modelo, "predict_proba"):
        y_proba = modelo.predict_proba(X_test_tfidf)[:,1]
    else:
        y_proba = modelo.predict(X_test_tfidf)
    
    # Melhor threshold pelo F1
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = f1_scores.argmax()
    best_t = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    
    y_pred = (y_proba >= best_t).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    comparativo.append({"Modelo": nome, "Accuracy": acc, "F1": f1, "Best Threshold": best_t})
    
    print(f"\n=== {nome} ===")
    print(f"Best threshold: {best_t:.2f}")
    print(classification_report(y_test, y_pred))
    
    # Atualiza melhor modelo
    if f1 > melhor_f1:
        melhor_f1 = f1
        melhor_modelo_nome = nome
        melhor_modelo_pred = y_pred
        melhor_threshold = best_t


# Tabelinha comparativa
df_comparativo = pd.DataFrame(comparativo).sort_values(by="F1", ascending=False)
print("\n=== Comparativo Final de Modelos ===")
print(df_comparativo)


# Matriz de confusão do melhor modelo
cm = confusion_matrix(y_test, melhor_modelo_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Matriz de Confusão - {melhor_modelo_nome}")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
