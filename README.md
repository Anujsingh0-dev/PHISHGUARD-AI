# PHISHGUARD AI 🛡️
### Machine Learning-Based Phishing URL and Website Detection System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-PhiUSIIL%20(UCI)-green)](https://archive.ics.uci.edu/dataset/967)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> **"Detect suspicious websites through intelligent machine learning."**

---

## 📌 Problem Statement

Phishing websites are fraudulent pages designed to trick users into revealing sensitive information — login credentials, banking details, personal data. Traditional blacklist-based approaches cannot identify the thousands of new phishing sites created daily.

This project explores how **supervised Machine Learning** can detect potentially phishing websites by learning patterns from 235,795 labelled examples in the official PhiUSIIL dataset.

> ⚠️ **Academic Prototype:** This system is for educational purposes only. It does not guarantee website safety and must never replace professional browser security tools.

---

## 🎯 Objectives

1. Demonstrate complete supervised ML pipeline on real phishing data.
2. Train and compare Logistic Regression, Decision Tree, and Random Forest.
3. Build a two-model architecture for honest, safe live predictions.
4. Create an interactive Streamlit web dashboard.
5. Present evaluation results transparently with proper metrics.

---

## 🏗️ Two-Model Architecture

This project uses **two separate models** — a deliberate design decision.

| Mode | Model | Input | Purpose |
|------|-------|-------|---------|
| **Mode A** | Full Dataset Model | All 54 PhiUSIIL features | Academic evaluation, research demo |
| **Mode B** | URL-Only Model | 22 URL-string features | Live prediction in web app |

**Why two models?**  
The full dataset uses webpage-derived features (HTML structure, form counts, SSL data) that cannot be extracted from a URL string without actually visiting the website — which would be a security risk. The URL-only model enables safe offline prediction using only the URL string.

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| **Name** | PhiUSIIL Phishing URL (Website) Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) |
| **Dataset ID** | 967 |
| **Instances** | 235,795 |
| **Features** | 54 |
| **Phishing** | 100,945 (Label = 0) |
| **Legitimate** | 134,850 (Label = 1) |
| **Authors** | Arvind Prasad, Shalini Chandra |
| **Donated** | March 2024 |
| **License** | CC BY 4.0 |

---

## 🤖 ML Algorithms

- **Logistic Regression** — Linear boundary, interpretable, fast baseline
- **Decision Tree Classifier** — Rule-based, visual, interpretable
- **Random Forest Classifier** — Ensemble of 100 trees, robust, best accuracy

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| ML Library | scikit-learn |
| Data Processing | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Web App | Streamlit |
| Dataset API | ucimlrepo |
| Serialisation | joblib |
| Testing | pytest |

---

## 📂 Project Structure

```
PHISHGUARD-AI/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
├── LICENSE                          # MIT License
│
├── data/
│   ├── raw/
│   │   └── phiusiil_dataset.csv     # Raw dataset (not committed to Git)
│   └── processed/                   # Preprocessed arrays (auto-generated)
│
├── scripts/
│   └── inspect_dataset.py           # EDA and visualisation script
│
├── notebooks/                       # Jupyter notebooks (optional)
│
├── src/
│   ├── __init__.py
│   ├── utils.py                     # Shared constants and helpers
│   ├── data_loader.py               # Dataset download and loading
│   ├── feature_extraction.py        # URL feature extraction (Mode B)
│   ├── preprocessing.py             # Full-dataset preprocessing pipeline
│   ├── train_full_model.py          # Mode A model training
│   ├── train_url_model.py           # Mode B URL-only model training
│   ├── evaluate_models.py           # Evaluation and report generation
│   └── predict.py                   # Prediction module (used by Streamlit)
│
├── models/
│   ├── full_dataset_model/          # Mode A artifacts (auto-generated)
│   └── url_only_model/              # Mode B artifacts (auto-generated)
│
├── reports/
│   ├── figures/                     # All charts and graphs (auto-generated)
│   └── metrics/                     # JSON evaluation results (auto-generated)
│
├── app/
│   └── app.py                       # Streamlit web application
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_feature_extraction.py
│   └── test_prediction.py
│
└── docs/
    ├── project_documentation.md
    └── viva_questions.md
```

---

## 🚀 Installation and Setup

### Prerequisites
- macOS (tested on Apple Silicon M2)
- Python 3.11 or higher
- VS Code (recommended)
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/PHISHGUARD-AI.git
cd PHISHGUARD-AI
```

### Step 2 — Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Download the Dataset

```bash
python src/data_loader.py
```

This automatically downloads the official PhiUSIIL dataset from UCI.  
If automatic download fails, visit [UCI Dataset Page](https://archive.ics.uci.edu/dataset/967) and place the CSV at `data/raw/phiusiil_dataset.csv`.

---

## ▶️ Execution Order

**Must be run in this order:**

```bash
# 1. Download and inspect the dataset
python src/data_loader.py
python scripts/inspect_dataset.py

# 2. Preprocess (creates train/test splits and fitting pipeline)
python src/preprocessing.py

# 3. Train full-dataset model (Mode A)
python src/train_full_model.py

# 4. Train URL-only model (Mode B)
python src/train_url_model.py

# 5. Run all tests
python -m pytest tests/ -v

# 6. Launch the Streamlit app
streamlit run app/app.py
```

---

## 🌐 Streamlit Application

The dashboard has 6 pages:

| Page | Description |
|------|-------------|
| 🏠 Home Dashboard | Dataset stats, model overview, architecture |
| 📊 Dataset Analysis | Class distribution, correlations, feature distributions |
| 🤖 Model Performance | Comparison table, confusion matrix, ROC curve, feature importance |
| 🔍 URL Detection | Live URL input and ML-based prediction |
| 🔬 Research Model Demo | Full-dataset model demonstration |
| ℹ️ About Project | Dataset citation, limitations, academic disclaimer |

---

## 📊 Model Evaluation (Actual Results)

Results are generated after training. See `reports/metrics/` for JSON files.

> All metrics are computed using the held-out **test set (20%)** with **phishing as the positive class**.

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run individual test file
python -m pytest tests/test_feature_extraction.py -v
python -m pytest tests/test_data_loader.py -v
python -m pytest tests/test_prediction.py -v
```

---

## ⚠️ Limitations

1. The URL-only model uses only 22 URL-string features — no webpage content.
2. Sophisticated phishing sites with clean-looking URLs may evade detection.
3. The system was trained on one dataset — generalisation is not guaranteed.
4. This is an academic prototype — not a production security system.
5. A "Legitimate" prediction does NOT mean the website is safe.

---

## 🔮 Future Scope

- Browser extension integration for real-time protection
- Webpage content analysis for improved accuracy
- Domain reputation database integration
- More recent phishing datasets
- SHAP-based explainability

---

## 📚 Dataset Citation

```
Arvind Prasad and Shalini Chandra. (2023).
PhiUSIIL Phishing URL (Website) Dataset [Dataset].
UCI Machine Learning Repository.
https://doi.org/10.24432/C58W39
```

Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## 📝 Academic Details

**Student:** Anuj Singh  
**Roll No:** 2555178  
**Branch:** B.Tech Information Technology  
**Section:** A1  
**Institution:** CGC College of Engineering, Landran, Mohali  
**Batch:** 2025–2029  

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

The PhiUSIIL dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by Arvind Prasad and Shalini Chandra.
