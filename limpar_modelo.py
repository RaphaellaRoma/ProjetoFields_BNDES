import os
import shutil

path = "Modelos/modelo_bert_salvo"

delete_patterns = [
    "checkpoint-",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "runs",
]

for item in os.listdir(path):
    full = os.path.join(path, item)
    if any(item.startswith(p) for p in ["checkpoint-"]) or item in delete_patterns:
        if os.path.isdir(full):
            shutil.rmtree(full)
            print("Apagado:", full)
        else:
            os.remove(full)
            print("Apagado:", full)
