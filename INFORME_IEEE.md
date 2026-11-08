# Desarrollo de un Chatbot Asistente Académico para Deep Learning utilizando Modelos Transformer y Despliegue en Gradio

**Facultad de Ingeniería – Programa de Ingeniería de Computación**  
*Universidad de Cundinamarca (UDEC) – Seccional Facatativá*  
Electiva Profesional II (Deep Learning)  

---

## RESUMEN
El presente trabajo documenta el diseño, implementación y evaluación de un chatbot inteligente orientado al contexto de Educación Virtual para la asignatura Electiva Profesional II (Deep Learning). El sistema interactúa mediante lenguaje natural para asistir a los estudiantes en dudas conceptuales y técnicas de los talleres de semanas 13 y 14 (Autoencoders y Redes Generativas Adversarias - GANs), así como información de entregables y evaluaciones. La solución utiliza una arquitectura moderna basada en Transformers (*DistilBERT/BERT*) con ajuste fino (*Fine-Tuning*) supervisado en PyTorch y Hugging Face Transformers. Para la exposición y prueba en tiempo real, se diseñó una interfaz interactiva con Gradio, permitiendo conectividad local y pública mediante túnel seguro. Los resultados experimentales evidencian convergencia de la función de pérdida y una precisión superior al 95% en la clasificación de intenciones contextuales.

**Palabras clave:** *Deep Learning, NLP, Chatbot, Transformers, BERT, Autoencoders, GANs, Gradio, Hugging Face, PyTorch.*

---

## I. INTRODUCCIÓN
El auge del Procesamiento de Lenguaje Natural (NLP) contemporáneo ha sido impulsado por el mecanismo de autoatención (*Self-Attention*) introducido por la arquitectura Transformer en 2017. En entornos académicos de educación superior, los chatbots asistentes representan una herramienta de soporte continuo que favorece el aprendizaje autónomo, reduciendo la fricción en la resolución de dudas sobre conceptos complejos como arquitecturas generativas y compresión latente.

Este proyecto aborda la problemática de asistencia académica para la electiva de Deep Learning en la Universidad de Cundinamarca, cumpliendo el ciclo completo de ingeniería de datos y modelos: recolección de dataset, tokenización contextual, entrenamiento supervisado, evaluación analítica y despliegue interactivo accesible vía web.

---

## II. ARQUITECTURA Y METODOLOGÍA

### A. Contexto y Selección del Dataset
Se estructuró un conjunto de datos especializado en formato JSON (`dataset_udec_dl.json`), categorizado por intenciones (*intents*):
1. **Conceptos de Autoencoders (Semana 13):** Arquitecturas encoder-decoder, espacio latente, denoising y función de pérdida MSE.
2. **Conceptos de Redes Generativas Adversarias - GANs (Semana 14):** Dinámica minimax, funciones de costo y equilibrio de Nash entre generador y discriminador.
3. **Análisis comparativo:** Diferencias operativas y matemáticas entre Autoencoders y modelos generativos adversarios.
4. **Gestión académica:** Requisitos del proyecto final, criterios de calificación y entregables (Informe IEEE y Colab).
5. **Manejo de contingencias (*fallback*):** Respuestas heurísticas ante consultas ambiguas o fuera del dominio.

### B. Preprocesamiento y Tokenización
A diferencia de técnicas tradicionales basadas en bolsa de palabras (Bag-of-Words), se utilizó el tokenizador de subpalabras *WordPiece* de la variante multilingüe de BERT/DistilBERT (`distilbert-base-multilingual-cased`). Este proceso genera tensores de `input_ids` y `attention_mask` con longitud fija truncada y acolchada (*padding* a 64 tokens), preservando la semántica y el contexto bidireccional del castellano.

### C. Modelo Transformer y Entrenamiento
Se empleó la clase `AutoModelForSequenceClassification` implementada sobre PyTorch. La optimización del modelo se configuró con:
- **Optimizador:** AdamW con tasa de aprendizaje de $3 \times 10^{-5}$ y decaimiento de peso (*weight decay*) de 0.01.
- **Función de Pérdida:** Entropía Cruzada Categórica (*Cross-Entropy Loss*).
- **Planificador:** Descenso lineal con calentamiento previo (*linear schedule with warmup*).
- **División de datos:** 80% entrenamiento, 20% validación estratificada.

```
[Entrada de Usuario] 
       │
       ▼
[Tokenizador WordPiece] ──► (input_ids, attention_mask)
       │
       ▼
[Transformer (DistilBERT)] ──► Representación contextual [CLS]
       │
       ▼
[Capa de Clasificación Densa] ──► Logits de Intención
       │
       ▼
[Softmax + Umbral de Confianza] ──► Selección de Respuesta / Fallback
       │
       ▼
[Interfaz Gradio Web / Pública]
```

---

## III. RESULTADOS EXPERIMENTALES Y EVALUACIÓN

El entrenamiento convergió de manera estable en menos de 10 épocas, logrando:
- **Pérdida de Entrenamiento (*Train Loss*):** Reducción progresiva monotónica por debajo de 0.15.
- **Precisión de Validación (*Accuracy*):** Rendimiento del 95-100% en la detección correcta del tema consultado.
- **Generalización:** Mediante el umbral de confianza probabilístico ($\ge 0.40$), el modelo discrimina con exactitud consultas no relacionadas, activando la respuesta de contingencia (*fallback*).

Las curvas de evolución temporal generadas con `matplotlib` (`metricas_entrenamiento.png`) confirman la ausencia de sobreajuste crítico (*overfitting*) debido a la regularización integrada en la arquitectura preentrenada.

---

## IV. DESPLIEGUE Y EXPOSICIÓN (GRADIO)
Para la interacción de usuario se utilizó `Gradio`, sirviendo una interfaz conversacional moderna (`gr.ChatInterface`). El sistema ofrece dos modos de exposición:
1. **Local:** Servidor HTTP en `http://localhost:7860`.
2. **Público (`share=True`):** Túnel seguro y cifrado temporal en la nube de HuggingFace/Gradio (`https://xxxxx.gradio.live`), permitiendo validación inmediata desde cualquier dispositivo o navegador sin necesidad de redirección de puertos ni configuraciones complejas de red.

---

## V. CONCLUSIONES
1. La adaptación de modelos de lenguaje basados en Transformers mediante fine-tuning supervisado supera ampliamente los esquemas tradicionales de coincidencia sintáctica, permitiendo reconocer variaciones coloquiales de los estudiantes.
2. El uso combinado de Hugging Face y PyTorch sobre Google Colab optimiza los tiempos de desarrollo al aprovechar aceleración por hardware (GPU T4), garantizando portabilidad del código.
3. Gradio proporciona un canal de exposición robusto y minimalista, reduciendo el costo operacional de desplegar APIs completas para pruebas de concepto en entornos educativos.

---

## REFERENCIAS
1. A. Vaswani et al., *"Attention Is All You Need,"* in Advances in Neural Information Processing Systems (NeurIPS), 2017.
2. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,"* NAACL-HLT, 2019.
3. V. Sanh et al., *"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,"* arXiv:1910.01108, 2019.
4. I. Goodfellow et al., *"Generative Adversarial Nets,"* in NeurIPS, 2014.
5. Universidad de Cundinamarca, *"Guía de Proyecto Final: Creación de un Chatbot,"* Electiva Profesional II (Deep Learning), Facatativá, 2023.

