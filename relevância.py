import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('normativos_processados.csv')
df = pd.DataFrame(data)
df = df[df["aplicavel_bndes"] == True]

X = df["texto_normativo"]
y = df["relevancia"]

# # Codificar rótulos
# le = LabelEncoder()
# y_enc = le.fit_transform(y)


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
# )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


vectorizer = TfidfVectorizer(
    max_features=10000,
    sublinear_tf=True     # suaviza termos muito frequentes
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

from sklearn.model_selection import RandomizedSearchCV
import numpy as np

hidden_layer_sizes=[]

for i in range (128, 200):
    for j in range (60,120):
        hidden_layer_sizes.append((i,j))
param_dist = {
    'hidden_layer_sizes': hidden_layer_sizes,
    'activation': ['relu', 'tanh'],
    'alpha': np.logspace(-6, -2, 5),  # de 1e-6 até 1e-2
    'learning_rate': ['constant', 'adaptive']
}

# rand_search = RandomizedSearchCV(
#     MLPClassifier(max_iter=1000, early_stopping=True, random_state=42),
#     param_distributions=param_dist,
#     n_iter=15,
#     scoring='f1_macro',
#     cv=3,
#     random_state=42
# )

# rand_search.fit(X_train_tfidf, y_train)
# print("Melhor configuração (nova busca):", rand_search.best_params_)


model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-04,
    max_iter=100,
#    early_stopping=True,
    random_state=42
)


model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("\nDistribuição das classes no treino:")
print(pd.Series(y_train).value_counts(normalize=True))

print("\nRelatório de classificação final (MLP otimizado):")
#print(classification_report(y_test, y_pred, target_names=le.classes_))
print(classification_report(y_test, y_pred))








from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(
    max_iter=2000,
    C=1.0,
    penalty="l2",
    solver="lbfgs",
    random_state=42
)

logreg.fit(X_train_tfidf, y_train)
y_pred_lr = logreg.predict(X_test_tfidf)

print("\nRelatório de classificação (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4, max_iter=300, random_state=42)
}

for name, clf in models.items():
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
