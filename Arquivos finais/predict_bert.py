# predict_bert.py

import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

SAVE_DIR = "./Modelos/modelo_bert_salvo"


def predict_texts(texts):
    """
    Recebe um texto ou lista de textos e retorna as previsões.
    """

    # Aceitar tanto string quanto lista de strings
    if isinstance(texts, str):
        texts = [texts]

    # Carregar tokenizer e modelo
    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
    model.eval()

    # label encoder
    le = joblib.load(f"{SAVE_DIR}/label_encoder.pkl")

    # Tokenizar
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )

    # Desativar gradiente para inferência
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # Decodificar classes
    decoded = le.inverse_transform(preds)

    return decoded



if __name__ == "__main__":
    exemplo = [
        "Dispõe sobre as regras aplicáveis ao regime de Facilitação do Acesso a Capital e de Incentivos a Listagens – FÁCIL no âmbito do mercado de capitais.(Publicada no DOU de 04.07.2025)",
        "Ofício Circular para comunicar ao mercado a respeito de atualizações no sistema FundosNet, com aprimoramentos nos formulários estruturados."
    ]

    pred = predict_texts(exemplo)
    print("\n Previsões:")
    for texto, p in zip(exemplo, pred):
        print(f"- \"{texto}\" → classe: {p}")
