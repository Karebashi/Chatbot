"""
Entrenamiento y Fine-Tuning de Transformer (BERT / DistilBERT)
Proyecto Final - Electiva Profesional II (Deep Learning) - UDEC Facatativa
"""

import json
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# 1. Cargar Dataset
DATASET_PATH = "dataset_udec_dl.json"
MODEL_NAME = "distilbert-base-multilingual-cased"
OUTPUT_DIR = "./model_checkpoint"
BATCH_SIZE = 8
EPOCHS = 12
LR = 3e-5

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts, labels = [], []
tag_to_id = {}
id_to_tag = {}

import re
import unicodedata

def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[¿?¡!.,:;()"\']', '', text)
    text = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in text if not unicodedata.combining(c))

for idx, intent in enumerate(data["intents"]):
    tag = intent["tag"]
    tag_to_id[tag] = idx
    id_to_tag[idx] = tag
    for pattern in intent["patterns"]:
        texts.append(clean_text(pattern))
        labels.append(idx)

# Guardar mapeo de clases
with open("label_mapping.json", "w", encoding="utf-8") as f:
    json.dump({"tag_to_id": tag_to_id, "id_to_tag": id_to_tag}, f, indent=2, ensure_ascii=False)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 2. Tokenizacion
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[item], dtype=torch.long)
        }

train_dataset = IntentDataset(train_texts, train_labels, tokenizer)
val_dataset = IntentDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 3. Cargar Modelo Transformer Preentrenado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de ejecucion: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(tag_to_id)
).to(device)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

# 4. Bucle de Entrenamiento y Evaluacion
train_losses, val_losses, val_accuracies = [], [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validacion
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            val_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total if total > 0 else 0
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoca {epoch + 1}/{EPOCHS} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

# 5. Guardar Checkpoint
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Modelo guardado exitosamente en {OUTPUT_DIR}")

# 6. Graficas de Rendimiento (Matplotlib) para el Informe IEEE
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", marker="o")
plt.plot(range(1, EPOCHS + 1), val_losses, label="Val Loss", marker="s")
plt.xlabel("Epocas")
plt.ylabel("Loss")
plt.title("Curva de Perdida (Loss)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), val_accuracies, label="Val Accuracy", color="green", marker="^")
plt.xlabel("Epocas")
plt.ylabel("Accuracy")
plt.title("Precision de Validacion")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("metricas_entrenamiento.png", dpi=300)
print("Graficas guardadas en metricas_entrenamiento.png")

