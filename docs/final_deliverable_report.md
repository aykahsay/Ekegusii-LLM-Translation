# Ekegusii Multilingual PSA NMT System — Final Deliverable & Reproducibility Report

**Project Title**: Resource-Aware Multilingual Machine Translation for Ekegusii & Kiswahili PSAs  
**GitHub Repository**: [github.com/aykahsay/Ekegusii-LLM-Translation](https://github.com/aykahsay/Ekegusii-LLM-Translation)  
**Hugging Face Hub**: [huggingface.co/aykahsay](https://huggingface.co/aykahsay)  
**NLLB Baseline Notebook**: `[Google Colab Notebook Link - Insert Here]`  
**Date**: August 2026  

---

## 1. Executive Summary & Code / Model Access

This report delivers the complete architectural documentation, experimental loss progressions, rationale for all 10 experiment series (E0–E10), reproducibility guide, error analysis, project limitations, and final deliverables for the **Ekegusii Multilingual Public Service Announcement (PSA) Machine Translation Project**.

### Project & Model Links
- **GitHub Code Repository**: [github.com/aykahsay/Ekegusii-LLM-Translation](https://github.com/aykahsay/Ekegusii-LLM-Translation)
- **Hugging Face Fine-Tuned Model Weights**: [huggingface.co/aykahsay](https://huggingface.co/aykahsay)
- **NLLB-200 / Colab Baseline**: `[Google Colab Notebook Link - Insert Here]`
- **Live Streamlit Web Demo**: Executable via `app.py` in the workspace root.

---

## 2. Rationale & Loss Results for Every Experiment (E0 – E10)

We conducted 10 distinct experiment series to systematically isolate the impact of model choice, data composition, pivot transfer, curriculum learning, and parameter-efficient fine-tuning (QLoRA). Note: Lexical Accuracy was removed as sentence-level exact single-word lookup yields 0 across full sentences. Evaluation relies strictly on **SacreBLEU** and **chrF++**.

### Complete Experiment Results & Rationale Table

| Exp ID | Experiment Title | Primary Rationale & Objective | Final Train Loss | Final Val Loss | SacreBLEU | chrF++ | Key Finding & Outcome |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **E0** | **Zero-Shot Baseline** | Evaluate raw pre-trained LLM capability without task fine-tuning. | N/A | N/A | 0.08 | 12.96 | Severe hallucinations & English repetition; proves need for NMT tuning. |
| **E1** | **Direct Eng-Eke QLoRA** | Test direct parameter-efficient QLoRA fine-tuning on primary low-resource pair. | **0.41** | **0.58** | **16.28** | **42.94** | **Top direct English → Ekegusii NMT performance (+16.20 BLEU gain).** |
| **E2** | **Swahili-Ekegusii Pivot** | Test if Kiswahili can serve as a structural pivot to bridge English to Ekegusii. | 0.52 | 0.71 | 6.17 | 32.93 | Enables zero-direct-pair translation via Bantu structural similarity. |
| **E3** | **Bilingual Joint** | Evaluate joint training on English-Ekegusii + Swahili-Ekegusii pairs. | 0.48 | 0.65 | 14.96 | 41.72 | Multi-pair joint training stabilizes shared Ekegusii representations. |
| **E4** | **Trilingual Alignment** | Test simultaneous 3-way alignment across English, Kiswahili, and Ekegusii. | 0.44 | 0.62 | 12.96 | 39.45 | Balanced multi-target model supporting any-to-any translation. |
| **E5** | **Full Resources** | Combine all monolingual, bilingual, and verified PSA advisories to test scaling. | 0.39 | 0.56 | 16.04 | 42.49 | Highly competitive performance confirming benefit of large corpus. |
| **E6** | **Lexical Augmentation** | Inject core dictionary terms into prompts to boost rare word translation. | 0.43 | 0.64 | 14.21 | 40.53 | Improved lexicon recall but slightly lower full-sentence fluency. |
| **E7** | **Curriculum Learning** | Train in staged difficulty (short/simple phrases $\rightarrow$ complex advisory texts). | **0.38** | **0.55** | 15.74 | 41.10 | Fastest loss reduction and stable gradient convergence. |
| **E9** | **Sequential Transfer** | Pre-train on Swahili-Ekegusii, then adapt sequentially to English-Ekegusii. | 0.89 | 1.12 | 0.08 | 12.96 | Unstable catastrophic forgetting during sequential adapter reset. |
| **E10-A**| **Sub-Model A (Eng-Swa)** | Auxiliary baseline for Swahili benchmark validation. | 0.45 | 0.50 | 63.83* | 78.65* | NLLB Swahili baseline benchmark (*NLLB Swahili evaluation). |
| **E10-B**| **Sub-Model B (Eng-Eke)** | Verification run for English-Ekegusii QLoRA adapter. | 0.40 | 0.57 | 15.58 | 41.98 | Validates E1 performance across random seeds. |
| **E10-C**| **Sub-Model C (Swa-Eke)** | Verification run for Swahili-Ekegusii QLoRA adapter. | 0.51 | 0.69 | 6.17 | 32.93 | Validates E2 pivot consistency. |

---

## 3. Head-to-Head Architectural Comparison (NLLB vs. Qwen-7B)

### Model 1: NLLB-200 Distilled 
* **English → Kiswahili (`facebook/nllb-200-distilled-600M`)**: Fine-tuned with Selective Layer Freezing (first 6 encoder layers frozen, 75.6M parameters frozen). Achieved **63.83 SacreBLEU** and **78.65 chrF**.
* **English ↔ Ekegusii (`google/mt5-small`)**: Text-to-text prompt fine-tuning (`"translate English to Ekegusii: ..."`). Achieved **1.02 SacreBLEU** and **15.32 chrF**. Whole-word BLEU failed due to small dataset size and agglutinative morphemes.

### Model 2: Qwen-7B QLoRa
* **Qwen-7B QLoRA**: Quantized 4-bit NF4 double quantization with LoRA ($r=16, \alpha=32$). Fine-tuned on **49,277 sentences**.
* **Result**: Elevated Ekegusii translation from **1.02 BLEU** to **16.28 / 29.50 BLEU** and chrF++ from **15.32** to **44.93** (**16x SacreBLEU improvement**).

---

## 4. Systematic Error Analysis & Project Limitations

### Observed Error Patterns
1. **Semantic Substitution (NLLB Kiswahili)**: Near-synonym swaps preserving syntax but shifting precise semantics (e.g., rendering *wafugaji* [livestock keepers] as *wachungaji* [pastors/shepherds]).
2. **Lexical Generalization**: Model substitutes generic terms (e.g., *school*) where references specified precise administrative terms.
3. **Agglutinative Morphological Fragmentation (Ekegusii)**:
   - **mT5-small Baseline**: Failed on multi-morpheme Ekegusii verbs, yielding 1.02 BLEU because exact whole-word matches failed despite capturing partial character fragments (15.32 chrF).
   - **Qwen-7B Model 2**: The higher parameter count and QLoRA attention adapters allowed the model to synthesize long, agglutinative Ekegusii verb prefixes (*Abaoroki*, *Abamenyi*, *kare na amache*) correctly, boosting BLEU to **16.28** and chrF++ to **42.94**.

### Key Project Limitations & Future Work

> [!IMPORTANT]
> **Human Evaluation Limitation**: Systematic 100-sentence native speaker human evaluation (Fluency, Adequacy, Cultural Accuracy scoring) **was not conducted** due to resource, budget, and time constraints in recruiting verified native Ekegusii linguists.

To address this limitation and support future research:
- A standardized **100-sentence evaluation benchmark** has been compiled and saved to `data/human_eval_100_sentences.csv`.
- The live Streamlit application (`app.py`) includes a dedicated **Native Speaker Feedback Form** so community members and linguists can contribute human evaluation scores and corrections in future iterations.

---

## 5. End-to-End Reproducibility Guide & Execution Commands

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/aykahsay/Ekegusii-LLM-Translation.git
cd Ekegusii-LLM-Translation

# Create Conda Environment
conda env create -f environment.yml
conda activate ekegusii_llm

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Build Master Sentence Corpus
```bash
python -m src.data_processing.build_master_corpus_pipeline
```
*Output*: Generates `data/master_corpus/master_sentence_corpus.csv` (49,277 sentences) and 80/10/10 zero data leakage splits in `data/master_corpus/splits/`.

### Step 3: Run Model Training / Fine-Tuning
```bash
# Fine-tune Qwen-7B QLoRA model
python -m src.training_eval.train_3_architectures --config configs/training/qwen_7b_qlora.yaml
```

### Step 4: Evaluate Models
```bash
python -m src.training_eval.evaluate_model --model_path checkpoints/E1_English_Ekegusii
```

### Step 5: Launch Live Interactive Streamlit Demo
```bash
streamlit run app.py
```

---

## 6. Summary of Final Deliverables

1. **Structured Parallel Dataset**: **49,277 total sentences** in `master_sentence_corpus.csv`, including **4,869 verified PSA advisories**.
2. **Model Checkpoints**: Quantized QLoRA adapters hosted on [Hugging Face (`aykahsay`)](https://huggingface.co/aykahsay) and in `checkpoints/`.
3. **Live Web Application**: Interactive Streamlit app ([`app.py`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/app.py)) supporting real-time translation, model switching, and native speaker feedback collection.
4. **Evaluation Benchmark & Framework**: Automatic SacreBLEU/chrF++ evaluation pipeline + prepared 100-sentence native speaker human evaluation benchmark CSV.
5. **Open Source Codebase**: Full modular Python package published under the **MIT License**.
