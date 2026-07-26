# 📢 Multilingual Public Service Announcement (PSA) Machine Translation for Kenya

<p align="center">
  <a href="https://colab.research.google.com/github/aykahsay/Multilogual_transaltion_nlp/blob/main/notebooks/colab_training.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <img src="https://img.shields.io/badge/License-CC--BY--4.0-blue.svg" alt="CC-BY-4.0 License"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-green.svg" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg" alt="Transformers"/>
  <img src="https://img.shields.io/badge/Streamlit-App-red.svg" alt="Streamlit"/>
</p>

An end-to-end **Multilingual Neural Machine Translation (NMT)** project designed to translate and evaluate official Public Service Announcements (PSAs) in Kenya between **English**, **Kiswahili**, and **Ekegusii (Gusii)**—a low-resource indigenous language.

---

## 🌟 Key Highlights & Dataset Overview

* 📊 **7,678 Confirmed PSA Parallel Sentences** across 5 national advisory domains:
  * 🩺 **Health & Disaster Relief**
  * 🔒 **Security & Public Safety**
  * 📚 **Education & Civic Rights**
  * 🌾 **Agriculture & Food Security**
  * ⚖️ **Governance & Institutional Notices**
* 🇰🇪 **Clean Parallel Datasets**:
  * **English $\leftrightarrow$ Kiswahili**: `5,752` parallel pairs (0 N/A placeholders)
  * **English $\leftrightarrow$ Ekegusii**: `4,557` parallel pairs (0 N/A placeholders)
  * **3-Way Trilingual Triplets**: `2,806` complete triplets
* 🤖 **Machine Learning Classifier**: TF-IDF + Logistic Regression model (**91.4% accuracy**) to verify if candidate text is a valid PSA.
* 🚀 **Google Colab Master Notebook**: Single GPU notebook ([`notebooks/colab_training.ipynb`](https://github.com/aykahsay/Multilogual_transaltion_nlp/blob/main/notebooks/colab_training.ipynb)) with self-contained, well-commented code chunks for **Meta NLLB-200** and **MarianMT** fine-tuning.
* 🎨 **Interactive Streamlit Web App**: Serves as a digital public good featuring single-sentence translation, preset advisories, confidence metrics, and native speaker evaluation forms.

---

## 📁 Repository Structure

```
Multilogual_transaltion_nlp/
├── app.py                                   # Streamlit Web Application
├── requirements.txt                         # Python project dependencies
├── psa_classifier.pkl                       # Trained TF-IDF PSA Logistic Regression Classifier
├── tfidf_vectorizer.pkl                     # TF-IDF Feature Vectorizer
├── README.md                                # Project Documentation
│
├── data/
│   ├── Master_Mixed_Data.csv                # Complete corpus (8,290 entries, PSA + Non-PSA)
│   ├── Master_PSA_Only.csv                  # Verified PSA corpus (7,678 parallel sentences)
│   ├── dataset_metadata.json                # Full metadata specification (CC-BY-4.0)
│   ├── human_eval_100_sentences.csv         # 100-sentence benchmark dataset for native evaluators
│   ├── final_data/                          # Clean final dataset outputs
│   │   ├── English_Swahili_Dataset.csv
│   │   ├── English_Ekegusii_Dataset.csv
│   │   ├── Trilingual_English_Swahili_Ekegusii_Dataset.csv
│   │   └── English_Domain_Dataset.csv
│   ├── languages/                           # Clean parallel datasets by language
│   │   ├── PSA_English_Swahili.csv
│   │   ├── PSA_English_Ekegusii.csv
│   │   └── PSA_Trilingual_Complete.csv
│   ├── raw/                                 # Raw PDFs, CSVs, and Twitter dumps
│   └── intermediate/                        # Legacy splits & intermediate preprocessing
│
├── notebooks/
│   ├── colab_training.ipynb                 # Master Google Colab GPU Notebook
│   └── kenya_treasury_scraper_and_translation.ipynb
│
├── src/
│   ├── scrapers/                            # Web & PDF scraping scripts
│   ├── data_processing/                     # Dataset compilation & language splitting
│   │   ├── build_master_datasets.py
│   │   ├── split_by_language.py
│   │   ├── generate_human_eval_dataset.py
│   │   └── generate_dataset_metadata.py
│   └── training_eval/                       # Model fine-tuning & evaluation scripts
│       ├── train_ekegusii.py
│       ├── train.py
│       └── evaluate.py
│
└── reports/
    └── Week_1_Data_Collection_and_Curation_Report.md
```

---

## ⚡ Quickstart & Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/aykahsay/Multilogual_transaltion_nlp.git
cd Multilogual_transaltion_nlp
pip install -r requirements.txt
```

### 2. Run Data Processing & Dataset Splitting
```bash
python src/data_processing/build_master_datasets.py
python src/data_processing/split_by_language.py
```

### 3. Launch the Interactive Web App
```bash
streamlit run app.py
```

---

## 🏋️‍♂️ Fine-Tuning & Model Training

### Training on Google Colab (Recommended)
Open the master GPU notebook directly in Google Colab:  
👉 **[Open Master Colab Notebook](https://colab.research.google.com/github/aykahsay/Multilogual_transaltion_nlp/blob/main/notebooks/colab_training.ipynb)**

### Training Locally via CLI
* **English ➡️ Ekegusii (Meta NLLB-200)**:
  ```bash
  python src/training_eval/train_ekegusii.py
  ```
* **English ➡️ Swahili (MarianMT)**:
  ```bash
  python src/training_eval/train.py
  ```
* **Automatic Evaluation (SacreBLEU & chrF)**:
  ```bash
  python src/training_eval/evaluate.py --model_path models/psa-en-sw-finetuned --target_lang Kiswahili
  ```

---

## 📊 Dataset Metadata & Provenance

Complete dataset metadata including domain distributions and source attribution is available in [`data/dataset_metadata.json`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20(USIU)/Documents/NLP/Multilogual_transaltion_nlp/data/dataset_metadata.json):

* **Data Sources (12 Organizations)**: Ministry of Health Kenya (@MOH_Kenya), Ministry of Agriculture, NDMA, DCI Kenya, KNCHR, TSC, KICD, NPS, IFRC / Red Cross, UNICEF Kenya, KEPHIS, and PSC.
* **License**: Creative Commons Attribution 4.0 International ([CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)).

---

## 📜 License
This repository is licensed under the **MIT License**. The underlying parallel dataset is distributed under **CC-BY-4.0**.
