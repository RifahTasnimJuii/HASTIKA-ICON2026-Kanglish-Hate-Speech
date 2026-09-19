# HASTIKA @ ICON-2026: Kannada-English Code-Mixed Hate Speech Detection

## 🏆 Overview
This repository contains the code and methodology for the **HASTIKA shared task** at ICON-2026, focusing on binary hate speech detection in low-resource Kannada-English (Kanglish) code-mixed text. 

Our approach leverages a **Novel Ensemble Framework** combining **MuRIL** (optimized for Indian code-mixed languages) and **XLM-RoBERTa Large** (for robust global contextual representations), trained with **Focal Loss** to handle class imbalance and maximize Macro-F1.

## 🛠️ Methodology
- **Data Preprocessing:** Transliteration normalization, HTML entity cleaning, and URL/Mention tokenization.
- **Models:** 
  - `google/muril-base-cased` (Focal Loss, Class-Weighted)
  - `xlm-roberta-large` (Focal Loss, Class-Weighted)
- **Ensemble Strategy:** Soft-voting (Probability Averaging) with calibrated decision thresholds.

## 📂 Repository Structure
- `model_training.py`: Contains the training pipeline with Focal Loss and Early Stopping.
- `ensemble_inference.py`: Generates the final submission file by ensembling MuRIL and XLM-R predictions.
- `requirements.txt`: Python dependencies.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/RifahTasnimJuii/HASTIKA-ICON2026-Kanglish-Hate-Speech.git
   cd HASTIKA-ICON2026-Kanglish-Hate-Speech

2. Install dependencies:
      pip install -r requirements.txt

3. Download the HASTIKA dataset from the official GitHub and place the CSV files in a data/ folder.

4. Run training (example):
      python model_training.py

📊 Results
Validation Macro-F1: ~0.80+ (Single Model)
Primary Metric: Macro-averaged F1-score

📝 Citation & Academic Use
This work is developed as part of an undergraduate thesis research on "Transliteration-Aware Ensemble Learning for Low-Resource Code-Mixed Hate Speech Detection".

📧 Contact
Author: Rifah Tasnim Jui
Email: tazfahh@gmail.com
LinkedIn: www.linkedin.com/in/rifah-tasnim-a32a42258
GitHub: https://github.com/RifahTasnimJuii

