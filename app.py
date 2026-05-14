import pandas as pd
import re
import spacy
import gradio as gr
import fitz
import pickle
import joblib

model = joblib.load('resume_model_compressed.pkl')

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

def clean_resume(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and len(token.text) > 2]
    return " ".join(tokens)

def predict_resume(text_input, file_input):
    raw_text = ""
    if file_input is not None:
        try:
            with fitz.open(file_input.name) as doc:
                for page in doc:
                    raw_text += page.get_text()
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    elif text_input and text_input.strip():
        raw_text = text_input
    else:
        return "Please upload a PDF or paste resume text first!"

    try:
        cleaned = clean_resume(raw_text)
        if not cleaned.strip():
            return "The resume text could not be processed. Is it a scanned image?"
  
        vec = tfidf.transform([cleaned])
        probs = model.predict_proba(vec)[0]
        categories = model.classes_

        results = {categories[i]: float(probs[i]) for i in range(len(categories))}
        return results
    except Exception as e:
        return f"Prediction Error: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Resume Category Predictor")
    gr.Markdown("Upload a PDF resume or paste the text to see the predicted job category.")
    
    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(label="Paste Resume Text", lines=5)
            file_in = gr.File(label="Or Upload PDF", file_types=[".pdf"])
            submit_btn = gr.Button("Analyze Resume", variant="primary")
        
        with gr.Column():
            label_out = gr.Label(label="Top Predictions", num_top_classes=3)

    submit_btn.click(fn=predict_resume, inputs=[text_in, file_in], outputs=label_out)

if __name__ == "__main__":
    demo.launch()