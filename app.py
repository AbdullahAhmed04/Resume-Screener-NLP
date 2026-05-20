import os
import re
import shutil
import tempfile
import zipfile
import pandas as pd
import fitz
import gradio as gr
import pickle
import joblib

model = joblib.load("model.pkl")

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

def clean_resume(text):
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_resume(text_input, file_input):
    raw_text = ""

    if file_input is not None:
        try:
            with fitz.open(file_input.name) as doc:
                for page in doc:
                    raw_text += page.get_text()
        except Exception as e:
            return {"Error": f"Could not read PDF: {str(e)}"}
    elif text_input and text_input.strip():
        raw_text = text_input
    else:
        return {"Input Required": 1.0}

    cleaned = clean_resume(raw_text)
    if not cleaned.strip():
        return {"The resume text could not be processed.": 1.0}

    vec = tfidf.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    categories = model.classes_

    results = {categories[i]: float(probs[i]) for i in range(len(categories))}
    return results

def batch_sort_resumes(file_objects):
    if not file_objects:
        return None, "Please upload at least one PDF file!"

    temp_dir = tempfile.mkdtemp()
    processed_count = 0
    errors = []
    category_counts = {}

    for file_obj in file_objects:
        file_path = file_obj.name
        original_name = os.path.basename(file_path)
        raw_text = ""
        
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    raw_text += page.get_text()
            
            cleaned = clean_resume(raw_text)
            if not cleaned.strip():
                errors.append(f"Skipped {original_name}: No text extracted.")
                continue
                
            vec = tfidf.transform([cleaned])
            
            predicted_category = model.predict(vec)[0] 
            safe_folder_name = "".join([c for c in predicted_category if c.isalpha() or c.isspace()]).strip()
            
            category_counts[safe_folder_name] = category_counts.get(safe_folder_name, 0) + 1
            
            category_path = os.path.join(temp_dir, safe_folder_name)
            os.makedirs(category_path, exist_ok=True)
            
            destination = os.path.join(category_path, original_name)
            shutil.copy(file_path, destination)
            processed_count += 1

        except Exception as e:
            errors.append(f"Error processing {original_name}: {str(e)}")

    if processed_count == 0:
        shutil.rmtree(temp_dir)  
        return None, f"Failed to process files.\nErrors:\n" + "\n".join(errors)

    zip_output_path = os.path.join(tempfile.gettempdir(), "Sorted_Resumes.zip")
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                full_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_file_path, temp_dir)
                zipf.write(full_file_path, relative_path)

    shutil.rmtree(temp_dir)

    status_lines = []
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        status_lines.append(f"{cat.upper()}: {count}")
        
    status_message = "\n".join(status_lines)
        
    if errors:
        status_message += f"\n\nErrors encountered:\n" + "\n".join(errors)

    return zip_output_path, status_message

with gr.Blocks(theme=gr.themes.Soft(primary_hue="pink", secondary_hue="zinc")) as demo:
    gr.Markdown("# AI Resume Screening Hub")
    gr.Markdown("An end-to-end NLP assistant to analyze individual candidates or sort massive applicant pools instantly.")
    
    with gr.Tab("Single Resume Predictor"):
        with gr.Row():
            with gr.Column():
                text_in = gr.Textbox(label="Paste Resume Text", lines=5, placeholder="Paste CV text string content directly...")
                file_in = gr.File(label="Or Upload PDF", file_types=[".pdf"])
                submit_btn = gr.Button("Analyze Resume", variant="primary")
            with gr.Column():
                label_out = gr.Label(label="Top Predictions", num_top_classes=3)
        submit_btn.click(fn=predict_resume, inputs=[text_in, file_in], outputs=label_out)

    with gr.Tab("Bulk Batch Sorting"):
        gr.Markdown("### Bulk Processing Pipeline")
        gr.Markdown("Drop a collection of multiple candidate PDFs here. The application will automatically isolate text features, predict job sectors, arrange them into categorized subfolders, and compile a structured ZIP archive for clean extraction.")
        
        with gr.Row():
            with gr.Column():
                batch_btn = gr.Button("Process & Sort Batch", variant="primary")
                
        with gr.Row():
            with gr.Column():
                batch_in = gr.File(label="Upload Resumes (PDF only)", file_count="multiple", file_types=[".pdf"])
            with gr.Column():
                status_out = gr.Textbox(label="Resume Count", lines=6, interactive=False)
                file_out = gr.File(label="Sorted (.ZIP)", interactive=False)

        batch_btn.click(fn=batch_sort_resumes, inputs=[batch_in], outputs=[file_out, status_out])

if __name__ == "__main__":
    demo.launch(share=True)
