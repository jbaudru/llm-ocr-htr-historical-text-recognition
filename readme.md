# 📜 Archival Transcription with LLMs: Can LLMs facilitate the OCR/HTR tasks for historical records? 📚

![MIT License](https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI_GPT--4o-purple?logo=openai&logoColor=white)
![Claude](https://img.shields.io/badge/LLM-Claude_Sonnet_3.5-yellow?logo=anthropic&logoColor=white)
![Keras](https://img.shields.io/badge/OCR-Keras-red?logo=keras&logoColor=white)

This repository accompanies our study on using **Large Language Models (LLMs)** like GPT-4o and Claude Sonnet 3.5 to transcribe historical handwritten documents in a tabular format. We compare their performance against traditional OCR/HTR systems: EasyOCR, Keras OCR, Pytesseract, and TrOCR. 

## 🔍 Key Highlights
- **Models Tested**: GPT-4o, Claude Sonnet 3.5, EasyOCR, Keras OCR, Pytesseract, TrOCR.
- **Metrics**: Character Error Rate (CER) and BLEU.
- **Findings**:
  - GPT-4o is best for **line-by-line transcription**.
  - Claude Sonnet 3.5 excels at **whole-scan transcription**.
  - BLEU better captures transcription quality compared to CER.
  
## 🚀 Quick Start

1. **Clone the repo**:
   ```bash
   git clone https://github.com/jbaudru/img-analysis_seorin_project
   cd img-analysis_seorin_project
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```


## 📁 Data
The repository includes:

- Historical document scans (images) and transcriptions.
- Ground truth transcriptions for evaluation.
- Sample model predictions.

## 📜 License
This project is licensed under the MIT License. (https://opensource.org/license/mit)