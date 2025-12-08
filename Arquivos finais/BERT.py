# fine_tune_bert_train.py
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
from sklearn.utils import resample

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    Trainer
)
from torch.nn import CrossEntropyLoss
from datasets import Dataset

# Configurações gerais
CSV_PATH = "../Dados CSV/df_assunto_limpo.csv"
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
SAVE_DIR = "../Modelos/modelo_bert_salvo"

SEED = 42
MAX_LEN = 256
NUM_EPOCHS = 30
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 4
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 16


def read_csv_with_encodings(path):
    for enc in ["utf-8", "latin-1", "ISO-8859-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError("Erro ao ler CSV com encodings comuns.")

df = read_csv_with_encodings(CSV_PATH)



df = df[(df["aplicavel_bndes"] == 'Sim') & (df["relevancia"] != '0')].reset_index(drop=True)


# Oversampling para balanceamento
max_count = df['relevancia'].value_counts().max()
dfs = []
for label in df['relevancia'].unique():
    subset = df[df['relevancia'] == label]
    dfs.append(resample(subset, replace=True, n_samples=max_count, random_state=SEED))
df = pd.concat(dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)

texts = df["assunto_normativo"].astype(str).tolist()
labels_text = df["relevancia"].astype(str).tolist()


# Label encoding + split
le = LabelEncoder()
y = le.fit_transform(labels_text)

train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, y, test_size=0.3, stratify=y, random_state=SEED
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=SEED
)

# Tokenização
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def encode(texts, labels):
    enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN)
    enc["labels"] = labels
    return Dataset.from_dict(enc)

train_dataset = encode(train_texts, train_labels)
val_dataset = encode(val_texts, val_labels)
test_dataset = encode(test_texts, test_labels)

train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")


# Modelo BERT
num_labels = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
)

# Congelar primeiras 5 camadas para acelerar treino
for name, param in model.named_parameters():
    if name.startswith("bert.embeddings") or any(f"encoder.layer.{i}" in name for i in range(5)):
        param.requires_grad = False


# Pesos das classes
class_counts = np.bincount(train_labels)
class_counts = np.where(class_counts == 0, 1, class_counts)
inv_freq = 1.0 / (class_counts + 0.01)
inv_freq = inv_freq / inv_freq.sum() * len(inv_freq)
class_weights = torch.tensor(inv_freq, dtype=torch.float)


# Trainer customizado
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        device = next(model.parameters()).device
        weights = class_weights.to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# Métricas
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# TrainingArguments
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=SEED,
    fp16=torch.cuda.is_available(),
    report_to="none",
    gradient_accumulation_steps=2,
    remove_unused_columns=False,
)

# Treinamento
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
)

trainer.train()
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
joblib.dump(le, os.path.join(SAVE_DIR, "label_encoder.pkl"))


eval_results = trainer.evaluate(test_dataset)
print("\n✅ Métricas finais (TEST SET):")
for k, v in eval_results.items():
    print(f"{k}: {v:.4f}")

preds_output = trainer.predict(test_dataset)
preds = np.argmax(preds_output.predictions, axis=-1)

print("\n📊 Classification report:")
print(classification_report(test_labels, preds, target_names=le.classes_, zero_division=0))

cm = confusion_matrix(test_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Matriz de Confusão")
plt.tight_layout()
plt.show()