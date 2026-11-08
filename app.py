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
AVATAR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assistant-avatar.svg")

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

engine_label = "Transformer (DistilBERT/BERT)" if use_transformer else "TF-IDF N-gramas (Modo rápido)"

custom_css = """
.gradio-container {
    max-width: 1080px !important;
    margin: 0 auto !important;
}
#chatbot-hero {
    padding: 1.5rem 1.75rem;
    margin: 1rem 0;
    border: 1px solid #bfdbfe;
    border-radius: 20px;
    background: linear-gradient(120deg, #eff6ff, #dbeafe);
    align-items: center;
}
#chatbot-hero, #chatbot-hero * {
    color: #172554 !important;
}
#chatbot-hero h1 {
    margin-bottom: 0.4rem;
    font-size: clamp(1.7rem, 4vw, 2.4rem);
}
#chatbot-hero p {
    color: #334155 !important;
}
#engine-status {
    padding: 0.75rem 1rem;
    border: 1px solid #cbd5e1;
    border-radius: 14px;
    background: #f8fafc;
    color: #1e293b !important;
}
#engine-status * {
    color: #1e293b !important;
}
#chat-panel {
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    overflow: hidden;
}
@media (max-width: 640px) {
    #chatbot-hero {
        padding: 1rem;
    }
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
)

with gr.Blocks(title="Asistente de Deep Learning | UDEC") as demo:
    with gr.Row(elem_id="chatbot-hero"):
        gr.Image(
            value=AVATAR_FILE,
            show_label=False,
            container=False,
            height=88,
            width=88,
            interactive=False,
        )
        gr.Markdown(
            """
# Asistente Universitario de Deep Learning
### Electiva Profesional II · UDEC Facatativá

Resuelve dudas de la asignatura, repasa conceptos y encuentra orientación para los talleres y el proyecto final.
            """
        )

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "**Temas**  \n"
                "Deep Learning · Redes neuronales · CNN · Autoencoders · GANs"
            )
        with gr.Column():
            gr.Markdown(
                f"**Motor activo**  \n{engine_label}",
                elem_id="engine-status",
            )

    with gr.Column(elem_id="chat-panel"):
        gr.ChatInterface(
            fn=respond,
            chatbot=gr.Chatbot(
                height=460,
                show_label=False,
                layout="bubble",
                avatar_images=(None, AVATAR_FILE),
                buttons=["copy_all"],
                placeholder="¡Hola! Pregúntame sobre Deep Learning, los talleres o el proyecto.",
            ),
            textbox=gr.Textbox(
                placeholder="Escribe tu pregunta aquí...",
                lines=1,
                max_lines=4,
                container=False,
                submit_btn="Enviar",
                stop_btn="Detener",
            ),
            examples=[
                "¿Qué es Deep Learning y en qué se diferencia de Machine Learning?",
                "¿Cómo funciona una CNN para clasificar imágenes?",
                "¿Qué debo hacer en el taller de Autoencoders de la semana 13?",
                "¿Qué diferencia hay entre un Autoencoder y una GAN?",
                "¿Cómo se relacionan el generador y el discriminador en una GAN?",
                "¿Qué es el sobreajuste y cómo puedo evitarlo?",
                "¿Qué métricas puedo usar para evaluar mi modelo?",
                "¿Cómo elijo y divido los datos para mi proyecto?",
                "¿Cuál es el horario conocido de la asignatura?",
                "¿Cuándo vence el proyecto final?",
            ],
            examples_label="Prueba con una pregunta",
            show_progress="hidden",
            fill_height=False,
            analytics_enabled=False,
        )

    gr.Markdown(
        "La información de fechas de entrega debe confirmarse en la guía o el aula virtual."
    )

if __name__ == "__main__":
    # share=True para generar URL publica HTTPS gratuita en segundos
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=theme,
        css=custom_css,
    )
