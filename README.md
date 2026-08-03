# 📢 Multilingual Public Service Announcement (PSA) Machine Translation for Kenya

<p align="center">
  <a href="https://colab.research.google.com/github/aykahsay/Multilogual_transaltion_nlp/blob/main/translation_model_last.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <img src="https://img.shields.io/badge/License-CC--BY--4.0-blue.svg" alt="CC-BY-4.0 License"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-green.svg" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg" alt="Transformers"/>
  <img src="https://img.shields.io/badge/Streamlit-App-red.svg" alt="Streamlit"/>
</p>

An end-to-end **Multilingual Neural Machine Translation (NMT)** research platform and system designed to translate and evaluate official Public Service Announcements (PSAs) in Kenya across **English**, **Kiswahili**, and **Ekegusii (Gusii)**—a low-resource indigenous language.

---

## 🔗 Cloud Resources & Google Drive Links

* 📂 **[Google Drive — Preprocessed Data & Parallel Corpus](https://drive.google.com/drive/folders/1-iV1k2A9Ytz_-8r_ocrKvWaViAAsHjtN?usp=drive_link)**
* 🧠 **[Google Drive — Trained Model Checkpoints](https://drive.google.com/drive/folders/1fYLEQfIVKR4e5zi_F9plAjazJ-RcIfcC)**
* 🚀 **[Google Colab — Model Training Notebook](https://colab.research.google.com/github/aykahsay/Multilogual_transaltion_nlp/blob/main/translation_model_last.ipynb)**

---

## 🌟 Key Highlights & Dataset Overview

* 📊 **110,000+ Clean Multi-Domain & PSA Sentences** across 5 national advisory domains:
  * 🩺 **Health & Disaster Relief** (Ministry of Health, WHO, ReliefWeb Kenya)
  * 🔒 **Security & Public Safety** (DCI Bulletins, IPOA, Community Policing)
  * 📚 **Education & Civic Rights** (KNEC Guidelines, African Storybooks Literature)
  * 🌾 **Agriculture & Food Security** (NDMA Drought Advisories, IFAD)
  * ⚖️ **Governance & Institutional Notices** (Public Service Commission, Local News)
* 🇰🇪 **Organized Training Folders (Zero Duplication)**:
  * **Trilingual Corpora (`data_train_tringual/`)**: `33,889` clean rows (`bibile.csv`: 30,971, `psa.csv`: 2,783, `stories.csv`: 135)
  * **Bilingual Corpora (`data_train_bilingual/`)**: `9,009` unique pairs (`English_Ekegusii_Web_News.csv`: 4,289, `English_Ekegusii_Bible.csv`: 4,586, `Swahili_Ekegusii_Dictionary.csv`: 134)
  * **Unilingual Datasets (`data_train_unilingual/`)**: `67,274` monolingual sentences (`english_unilingual.csv`: 57,045, `swahili_unilingual.csv`: 5,910, `ekegusii_unilingual.csv`: 4,319)
  * **Reference Data (`ref_data/`)**: 10 clean reference datasets including ReliefWeb Kenya (47,628 rows), FineWeb Ekegusii web corpus (389 pages), Egesa FM web news (4,292 sentences), and Glosbe dictionaries.
* 🤖 **Domain Verification Classifier**: TF-IDF + Logistic Regression model (**91.4% accuracy**) to verify whether candidate text qualifies as an authentic PSA.
* 🚀 **Google Colab Master Model Training Notebook**: [`translation_model_last.ipynb`](https://github.com/aykahsay/Multilogual_transaltion_nlp/blob/main/translation_model_last.ipynb) for **Meta NLLB-200** and **MarianMT** fine-tuning using Low-Rank Adaptation (LoRA).
* 🎨 **Interactive Streamlit Web App**: Complete portal (`app.py`) for translation, batch CSV classification, metric dashboards, parallel corpus exploration, and native speaker human evaluation forms.

---

## 📊 Benchmark Evaluation Results

| Language Pair | Strategy / Model | SacreBLEU | chrF | COMET | Status / Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **English to Kiswahili** | NLLB-200 600M + LoRA (`r=16`) | **62.16** | **77.95** | **0.8120** | High-precision baseline on Bantu grammar |
| **English to Ekegusii** | Few-Shot Cross-Lingual Transfer | **4.09** | **30.04** | **0.7450** | Sub-word transfer verified; retains Ekegusii roots & stems |

---

## 🛠️ Diagnostic Troubleshooting & Cross-Lingual Transfer

During few-shot transfer for Ekegusii, an initial model collapse occurred (BLEU `0.74` / chrF `12.66`) due to infinite repetition loops. The following fixes were applied:
1. **Target Language Tag Override**: Forced sequence starter token `swh_Latn` to align with the pre-trained Bantu adapter layer, resolving `<unk>` token generation for `gus_Latn`.
2. **Constrained Beam Decoding**: Enabled Beam Search (`num_beams=4`), repetition penalty (`1.2`), and 3-gram repeat blockades (`no_repeat_ngram_size=3`).

---

## 📁 Repository Structure

```
Multilogual_transaltion_nlp/
├── app.py                                   # Streamlit Web Application Portal
├── requirements.txt                         # Python project dependencies
├── psa_classifier.pkl                       # Serialized TF-IDF Logistic Regression Classifier
├── tfidf_vectorizer.pkl                     # Serialized TF-IDF Feature Vectorizer
├── translation_model_last.ipynb            # Master Model Training Google Colab Notebook
├── README.md                                # Project Documentation & Overview
│
├── data_train_tringual/                     # Clean Trilingual Training Corpora (33,889 rows)
│   ├── bibile.csv                           # (30,971 rows) English, Kiswahili, Ekegusii, Religion
│   ├── psa.csv                              # (2,783 rows) English, Kiswahili, Ekegusii, Public Services
│   └── stories.csv                          # (135 rows) English, Kiswahili, Ekegusii, Literature
│
├── data_train_bilingual/                    # Unique Bilingual Corpora (9,009 rows)
│   ├── English_Ekegusii_Web_News.csv        # (4,289 rows) Real Egesa FM web news broadcasts
│   ├── English_Ekegusii_Bible.csv           # (4,586 rows) Unique English-Ekegusii Bible pairs
│   └── Swahili_Ekegusii_Dictionary.csv      # (134 rows) Glosbe & curated lexicon pairs
│
├── data_train_unilingual/                   # Unilingual Monolingual Datasets (67,274 rows)
│   ├── english_unilingual.csv               # (57,045 rows) English PSAs, ReliefWeb, NDMA
│   ├── swahili_unilingual.csv               # (5,910 rows) Swahili PSAs & Health advisories
│   └── ekegusii_unilingual.csv              # (4,319 rows) Ekegusii PSAs & FineWeb web pages
│
└── ref_data/                                # Complete Reference Datasets Collection (10 files)
    ├── ReliefWeb_Kenya_Disaster_PSAs_Raw.csv      # (47,628 rows)
    ├── Web_News_RMS_EgesaFM_English_Ekegusii.csv   # (4,292 rows)
    ├── Scraped_Government_PSAs_Verified.csv       # (1,863 rows)
    ├── NDMA_Drought_Advisories_English.csv        # (1,429 rows)
    ├── FineWeb_Ekegusii_Web_Corpus.csv            # (389 pages)
    ├── Ministry_of_Health_Social_Posts.csv         # (223 rows)
    ├── African_Storybooks_Multilingual_Corpus.csv # (135 rows)
    ├── Human_Evaluation_Benchmark_100.csv          # (100 rows)
    ├── Structured_Swahili_Ekegusii_Dictionary.csv # (84 rows)
    └── Online_Glosbe_Swahili_Ekegusii_Dictionary.csv # (50 rows)
```

---

## ⚡ Quickstart & Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/aykahsay/Multilogual_transaltion_nlp.git
cd Multilogual_transaltion_nlp
pip install -r requirements.txt
```

### 2. Access Project Datasets
All clean datasets are organized in their respective folders:
- `data_train_tringual/`: Trilingual parallel training sets
- `data_train_bilingual/`: Unique bilingual parallel translation pairs
- `data_train_unilingual/`: Language-specific monolingual corpora
- `ref_data/`: Reference collections and raw scraped sources

### 3. Run Model Training (Google Colab)
Open the master training notebook directly in Colab:  
👉 **[Open `translation_model_last.ipynb` in Colab](https://colab.research.google.com/github/aykahsay/Multilogual_transaltion_nlp/blob/main/translation_model_last.ipynb)**

### 4. Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```

---

## 📜 License & Citation
* **Repository License**: MIT License
* **Dataset License**: Creative Commons Attribution 4.0 International ([CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/))
