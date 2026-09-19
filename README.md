# HASTIKA @ ICON-2026: Kannada-English Code-Mixed Hate Speech Detection

## 🏆 Overview
This repository contains the code and methodology for the **HASTIKA shared task** at ICON-2026, focusing on binary hate speech detection in low-resource Kannada-English (Kanglish) code-mixed text. 

Our approach leverages a **Novel Ensemble Framework** combining **MuRIL** (optimized for Indian code-mixed languages) and **XLM-RoBERTa Large** (for robust global contextual representations), trained with **Focal Loss** to handle class imbalance and maximize Macro-F1.

## 🛠️ Methodology
- **Data Preprocessing:** Transliteration normalization, HTML entity cleaning, and URL/Mention tokenization.
- **Models:** 
  - `google/muril-base-cased` (Fine-tuned with Focal Loss & Class-Weighting)
  - `xlm-roberta-large` (Fine-tuned with Focal Loss & Class-Weighting)
- **Ensemble Strategy:** Soft-voting (Probability Averaging) with calibrated decision thresholds.

## 📂 Repository Structure
- `main_pipeline.py`: The complete end-to-end pipeline containing data preprocessing, dual-model training (MuRIL + XLM-R), and ensemble inference.
- `requirements.txt`: Python dependencies required to run the code.
- `data/`: (Not included) Directory where the HASTIKA dataset CSV files should be placed.

## 🚀 Quick Start
For users who want to run the entire pipeline (Training + Ensemble Inference) in one go:

```bash
python main_pipeline.py

🛠️ Detailed Setup & Execution
If you prefer to run the steps individually or set up the environment from scratch:
1. Clone the repository:
   git clone https://github.com/RifahTasnimJuii/HASTIKA-ICON2026-Kanglish-Hate-Speech.git
   cd HASTIKA-ICON2026-Kanglish-Hate-Speech

2. Install dependencies:
   pip install -r requirements.txt

3. Prepare the dataset:
   Download the HASTIKA dataset from the official GitHub and place the CSV files (binary_train.csv, binary_validation_inputs.csv, etc.) inside a data/ folder.

4. Run the pipeline:
   python main_pipeline.py
(Note: Ensure you have mounted your Google Drive or have sufficient GPU memory if running locally. The script is optimized for Google Colab.)

📊 Results
Validation Macro-F1: ~0.80+ (Single Model Baseline)
Ensemble Target Macro-F1: 0.85+ (MuRIL + XLM-RoBERTa Soft Voting)
Primary Metric: Macro-averaged F1-score

📝 Citation & Academic Use
This work is developed as part of an undergraduate thesis research on "Transliteration-Aware Ensemble Learning for Low-Resource Code-Mixed Hate Speech Detection".
If you use this code or methodology in your research, please cite it as:
@misc{jui2026hastika,
  author = {Rifah Tasnim Jui},
  title = {Transliteration-Aware Ensemble Learning for Low-Resource Code-Mixed Hate Speech Detection},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/RifahTasnimJuii/HASTIKA-ICON2026-Kanglish-Hate-Speech}
}

📧 Contact
Author: Rifah Tasnim Jui
Email: tazfahh@gmail.com
LinkedIn: linkedin.com/in/rifah-tasnim-a32a42258
GitHub: github.com/RifahTasnimJuii
