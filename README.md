# Proyecto Final: Chatbot Académico Deep Learning (UDEC Facatativá)

Chatbot inteligente para asistencia académica en la asignatura **Electiva Profesional II (Deep Learning)**, orientado a responder consultas sobre los talleres de las **Semanas 13 y 14 (Autoencoders y GANs)**, diferencias teóricas y lineamientos del proyecto final.

Basado en modelos **Transformers (BERT / DistilBERT)**, PyTorch y desplegado mediante **Gradio**.

---

## Archivos del Proyecto

1. **[Proyecto_Final_Chatbot_DeepLearning.ipynb](file:///c:/Users/Usuario/OneDrive/Desktop/Yo/UDEC/Noveno%20semestre/DL/Chatbot/Proyecto_Final_Chatbot_DeepLearning.ipynb)**: Cuaderno completo listo para subir y ejecutar directamente en **Google Colab** (entrenamiento + gráficas + Gradio `share=True`).
2. **[INFORME_IEEE.md](file:///c:/Users/Usuario/OneDrive/Desktop/Yo/UDEC/Noveno%20semestre/DL/Chatbot/INFORME_IEEE.md)**: Estructura y redacción completa del informe final en formato estándar IEEE (Resumen, Introducción, Metodología, Resultados, Conclusiones y Referencias).
3. **[dataset_udec_dl.json](file:///c:/Users/Usuario/OneDrive/Desktop/Yo/UDEC/Noveno%20swwwd de Autoencoders, GANs, horarios, entregables y fallbacks.
4. **[train_transformer.py](file:///c:/Users/Usuario/OneDrive/Desktop/Yo/UDEC/Noveno%20semestre/DL/Chatbot/train_transformer.py)**: Script de fine-tuning de DistilBERT con PyTorch, cálculo de métricas y generación de curvas en Matplotlib.
5. **[app.py](file:///c:/Users/Usuario/OneDrive/Desktop/Yo/UDEC/Noveno%20semestre/DL/Chatbot/app.py)**: Servidor de inferencia con interfaz Gradio (admite modo Transformer o fallback rápido TF-IDF si aún no se ha entrenado el checkpoint).
6. **[requirements.txt](file:///c:/Users/Usuario/OneDrive/Desktop/Yo/UDEC/Noveno%20semestre/DL/Chatbot/requirements.txt)**: Dependencias del proyecto.

---

## Cómo Ejecutar en Google Colab

1. Entra a [Google Colab](https://colab.research.google.com/).
2. Sube el archivo `Proyecto_Final_Chatbot_DeepLearning.ipynb`.
3. Activa el entorno de ejecución con GPU (Entorno de ejecución > Cambiar tipo de entorno > T4 GPU).
4. Ejecuta todas las celdas. La celda final generará una URL pública `https://xxxxx.gradio.live` para probar el bot de inmediato.

---

## Cómo Ejecutar en Local

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. (Opcional) Entrenar el Transformer en local
python train_transformer.py

# 3. Iniciar la aplicación
python app.py
```
Abre en tu navegador `http://localhost:7860`.
Para exponer públicamente desde tu máquina, activa `share=True` en la última línea de `app.py`.
