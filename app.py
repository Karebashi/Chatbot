import os
import json
import random
import unicodedata
import gradio as gr
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_FILE = "dataset_udec_dl.json"
CHECKPOINT_DIR = "./model_checkpoint"
LABEL_MAP_FILE = "label_mapping.json"

# Cargar dataset
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

responses = {intent["tag"]: intent["responses"] for intent in data["intents"]}
fallback_responses = data.get("fallback", ["Consulta fuera de alcance."])

# Intentar cargar Transformer entrenado
use_transformer = False
if os.path.exists(CHECKPOINT_DIR) and os.path.exists(LABEL_MAP_FILE):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
        model.eval()
        with open(LABEL_MAP_FILE, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            id_to_tag = {int(k): v for k, v in mapping["id_to_tag"].items()}
        use_transformer = True
        print("[INFO] Motor activo: Transformer BERT/DistilBERT fine-tuned.")
    except Exception as e:
        print(f"[WARN] Error cargando Transformer ({e}). Activando fallback TF-IDF.")
        use_transformer = False

# Motor secundario ligero (TF-IDF + Coseno)
import re

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[¿?¡!.,:;()"\']', '', text)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))

patterns, tags = [], []
for intent in data["intents"]:
    for p in intent["patterns"]:
        patterns.append(normalize(p))
        tags.append(intent["tag"])

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
pattern_vectors = vectorizer.fit_transform(patterns)

def predict_transformer(text: str):
    inputs = tokenizer(text, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        max_prob, pred_id = torch.max(probs, dim=0)

    if max_prob.item() < 0.25:
        return random.choice(fallback_responses)

    tag = id_to_tag.get(pred_id.item(), None)
    if tag and tag in responses:
        return random.choice(responses[tag])
    return random.choice(fallback_responses)

def predict_tfidf(text: str):
    q_vec = vectorizer.transform([normalize(text)])
    sims = cosine_similarity(q_vec, pattern_vectors)[0]
    best_idx = sims.argmax()
    if sims[best_idx] < 0.30:
        return random.choice(fallback_responses)
    best_tag = tags[best_idx]
    return random.choice(responses[best_tag])

def respond(message: str, history):
    if not message.strip():
        return "Escribe una consulta sobre la asignatura o los talleres."
    if use_transformer:
        return predict_transformer(message)
    return predict_tfidf(message)

engine_label = "Transformer (DistilBERT/BERT)" if use_transformer else "TF-IDF N-gramas (Modo rapido)"

demo = gr.ChatInterface(
    fn=respond,
    title="Asistente Virtual - Deep Learning UDEC Facatativá",
    description=f"Motor activo: **{engine_label}**.\nConsultas sobre talleres 13 y 14 (Autoencoders y GANs), horarios y entregables.",
    examples=[
        "¿De qué trata el taller de autoencoders de la semana 13?",
        "¿Cómo funciona el generador y discriminador en una GAN?",
        "¿Cuáles son las diferencias entre Autoencoders y GANs?",
        "¿Qué entregables tiene el proyecto final?",
        "¿Cómo se califica el proyecto?"
    ]
)

if __name__ == "__main__":
    # share=True para generar URL publica HTTPS gratuita en segundos
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
