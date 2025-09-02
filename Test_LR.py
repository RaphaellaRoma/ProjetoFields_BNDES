import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve

data = pd.read_csv('normativos_processados.csv')

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["texto_normativo"], df["aplicavel_bndes"], test_size=0.3, random_state=42, stratify=df["aplicavel_bndes"]
)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_proba = model.predict_proba(X_test_tfidf)[:, 1]




y_pred = model.predict(X_test_tfidf)


precision, recall, thresholds = precision_recall_curve(y_test, y_proba)



f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]

print(f"Melhor threshold pelo F1: {best_threshold:.2f}")
print(f"Precision: {precision[best_idx]:.2f}, Recall: {recall[best_idx]:.2f}, F1: {f1_scores[best_idx]:.2f}")


threshold = 0.47
y_pred_adj = (y_proba >= threshold).astype(int)

print('threshold ajustado:',classification_report(y_test, y_pred_adj))

print('sem ajuste:',classification_report(y_test, y_pred))