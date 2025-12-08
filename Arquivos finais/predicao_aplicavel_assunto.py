import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

SAVE_DIR_NN = "../Modelos/modelo_nn_salvo"

# --- Carregar artefatos na inicialização do módulo ---
try:
    vectorizer = joblib.load(f"{SAVE_DIR_NN}/vectorizer.pkl")
    model = joblib.load(f"{SAVE_DIR_NN}/model.pkl")
    threshold = joblib.load(f"{SAVE_DIR_NN}/threshold.pkl")
    print("Modelo NN (MLP, Vetorizador, Threshold) carregado.")
except FileNotFoundError:
    print(f"Erro: Arquivos de modelo não encontrados em {SAVE_DIR_NN}")
    print("Execute o script 'train_nn.py' primeiro.")
    vectorizer = None
    model = None
    threshold = 0.5 # Valor padrão de fallback

def predict_applicability(text: str) -> bool:
    """
    Prevê se um texto é aplicável (Sim/Não) usando o modelo MLP.
    Retorna True para 'Aplicável' (Sim) e False para 'Não Aplicável' (Não).
    """
    if not model or not vectorizer:
        print("Erro: Modelo NN não está carregado.")
        return False # Falha segura

    # O vetorizador espera uma lista (iterável) de textos
    text_list = [text]

    # 1. Vetorizar o texto
    text_tfidf = vectorizer.transform(text_list)

    # 2. Obter probabilidades da classe 1 ('Sim')
    y_proba = model.predict_proba(text_tfidf)[:, 1]

    # 3. Aplicar threshold
    # y_proba[0] é a prob do primeiro (e único) texto
    is_applicable = (y_proba[0] >= threshold)

    return bool(is_applicable)

if __name__ == "__main__":
    # Teste rápido
    exemplo_sim = "Projeto de energia renovável com foco em sustentabilidade."
    exemplo_nao = "Assunto aleatório sobre esportes."

    print(f"Testando '{exemplo_sim[:20]}...': {predict_applicability(exemplo_sim)}")
    print(f"Testando '{exemplo_nao[:20]}...': {predict_applicability(exemplo_nao)}")