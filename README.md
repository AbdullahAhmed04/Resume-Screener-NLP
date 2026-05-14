# AI Resume Category Predictor

![App Interface](app_screenshot.png)

An end-to-end NLP application that automatically categorizes resumes into 24 job sectors. Built with Python and deployed via a Gradio web interface.

## Project Overview
Manual resume screening is a bottleneck in recruitment. This tool uses Machine Learning to "read" resumes and instantly classify them into departments (e.g., Data Science, Healthcare, Arts, etc.), helping HR teams prioritize candidates efficiently.

## Technical Stack
*   **NLP:** Spacy (Lemmatization & Preprocessing), Regex
*   **Machine Learning:** Random Forest Classifier
*   **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
*   **UI/Frontend:** Gradio
*   **PDF Extraction:** PyMuPDF (fitz)
*   **Model Compression:** Joblib

## Key Features
*   **Multi-Format Support:** Paste raw text or upload a `.pdf` file.
*   **Advanced Cleaning:** Automatically removes URLs, emails, special characters, and "stop words."
*   **Probability Analysis:** Shows the top 3 most likely job categories with confidence scores.
*   **Optimized Performance:** Uses a compressed `joblib` model for fast inference.

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/AbduulahAhmed04/Resume-Screener-NLP.git](https://github.com/AbdullahAhmed04/Resume-Screener-NLP.git)
   cd Resume-Screener-NLP

2. **Install Dependencies:**
   pip install -r requirements.txt

3. **Download the Spacy Model:**
   python -m spacy download en_core_web_sm

4. **Launch the App:**
   python app.py


## Dataset & Accuracy
The model was trained on a Kaggle dataset of 2,400+ resumes. 
*   **Algorithm:** Random Forest
*   **Number of Categories:** 24
*   **Accuracy:** 72.23% (Random Forest Classifier)

## License
Distributed under the MIT License. See `LICENSE` for more information.
