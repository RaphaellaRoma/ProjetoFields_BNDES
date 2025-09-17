import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier



df = pd.read_csv("normativos_processados.csv")
df = df[df["aplicavel_bndes"] == True]

X = df["texto_normativo"].astype(str)
y = df["relevancia"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Vetorização TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

modelos = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64),activation="relu", solver="adam", alpha=1e-4, max_iter=300, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost" : XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42,
                              eval_metric="mlogloss", use_label_encoder=False)
    }



# Função matriz de confusão
def plot_confusion_matrix(y_true, y_pred, labels, titulo):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Matriz de Confusão - {titulo}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()


comparativo = []
melhor_f1 = -1
melhor_modelo_nome = ""
melhor_modelo_pred = None

for nome, modelo in modelos.items():
    print(f"\n=== {nome} ===")
    
    if nome == "XGBoost":
        modelo.fit(X_train_tfidf, y_train_enc)
        y_pred_enc = modelo.predict(X_test_tfidf)
        y_pred = le.inverse_transform(y_pred_enc)
        y_true = le.inverse_transform(y_test_enc)
    else:
        modelo.fit(X_train_tfidf, y_train)
        y_pred = modelo.predict(X_test_tfidf)
        y_true = y_test
    
    # Relatório
    print(classification_report(y_true, y_pred))
    
    # Métricas para comparativo
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    comparativo.append({"Modelo": nome, "Accuracy": acc, "F1 Macro": f1_macro})
    
    # Melhor modelo
    if f1_macro > melhor_f1:
        melhor_f1 = f1_macro
        melhor_modelo_nome = nome
        melhor_modelo_pred = y_pred
        melhor_modelo_true = y_true



# Tabelinha comparativa
df_comparativo = pd.DataFrame(comparativo).sort_values(by="F1 Macro", ascending=False)
print("\n=== Comparativo Final ===")
print(df_comparativo)


# Matriz de confusão do melhor modelo
plot_confusion_matrix(melhor_modelo_true, melhor_modelo_pred, labels=sorted(y.unique()), titulo=melhor_modelo_nome)
