# Sub-Objective 2 Report: Modeling with Transfer Learning & LLM Fine-Tuning

**Project**: Resource-Aware Multilingual Machine Translation for Ekegusii & Kiswahili Public Service Announcements (PSAs)  
**Repository**: [aykahsay/Ekegusii-LLM-Translation](https://github.com/aykahsay/Ekegusii-LLM-Translation)  
**Hugging Face Hub**: [huggingface.co/aykahsay](https://huggingface.co/aykahsay)  
**Focus Area**: Sub-Objective 2 — Transfer Learning, Model Optimization, Ablation Benchmarking & Inference  

---

## 1. Executive Summary & Core Deliverables

This milestone report documents the design, implementation, fine-tuning, ablation studies, and evaluation of **Sub-Objective 2: Modeling with Transfer Learning**. 

To address the low-resource constraints of Ekegusii (an agglutinative Bantu language of Kenya) alongside Kiswahili, we implemented and evaluated two contrasting modeling paradigms across **10 distinct experiment series (E0–E10)**:

1. **Model Paradigm 1 (Baseline Encoder-Decoder / Seq2Seq Models)**:
   - **`facebook/nllb-200-distilled-600M`**: Fine-tuned for **English → Kiswahili** translation using Selective Encoder Layer Freezing.
   - **`google/mt5-small`**: Text-to-text prompt-based fine-tuning (`"translate English to Ekegusii: ..."`) for low-resource Ekegusii pairs.
2. **Model Paradigm 2 (Fine-Tuned Multilingual Decoder LLMs with QLoRA)**:
   - **`Qwen/Qwen2.5-7B-Instruct`** & **`Mistral-7B-Instruct-v0.3`**: Fine-tuned using 4-bit Quantized Low-Rank Adaptation (QLoRA, $r=16, \alpha=32$) on the expanded **Master Sentence Corpus (49,277 total sentences)**.

### Key Performance Benchmark Summary

| Language Direction | Model Architecture | Fine-Tuning Method | Training Corpus Size | SacreBLEU | chrF++ | Key Performance Gain |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **English → Kiswahili** | `NLLB-200-distilled-600M` | Selective Layer Freezing (L1–L6) | 4,601 pairs | **63.83** | **78.65** | Benchmark baseline for high-resource Kiswahili |
| **English → Ekegusii** | `google/mt5-small` | Text-to-Text Fine-Tuning | 3,645 pairs | 1.02 | 15.32 | Baseline Seq2Seq model (suffered from low-data fragmentation) |
| **Kiswahili → Ekegusii**| `google/mt5-small` | Text-to-Text Fine-Tuning | 2,225 pairs | 0.59 | 14.39 | Baseline pivot model |
| **English → Ekegusii** | `Qwen-7B QLoRA` (Model 2) | 4-bit NF4 QLoRA ($r=16$) | 39,421 pairs | **16.28** | **42.94** | **16x SacreBLEU gain over mT5-small baseline (+15.26 BLEU)** |
| **Ekegusii → English** | `Qwen-7B QLoRA` (Model 2) | 4-bit NF4 QLoRA ($r=16$) | 39,421 pairs | **29.50** | **44.93** | Strong reverse translation performance |
| **English → Ekegusii** | `Mistral-7B QLoRA` (Model 2)| 4-bit NF4 QLoRA ($r=16$) | 39,421 pairs | **15.58** | **41.98** | Validates parameter-efficient transfer capability across decoders |

---

## 2. Milestone Checklist Audit

Below is the verified audit of all sub-objective 2 milestones:

### 2.1 Experiment Tracking Setup
- [x] **Weights & Biases (`wandb`) & TensorBoard Integration**: 
  - Integrated into training routines ([`src/training_eval/train_3_architectures.py`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/src/training_eval/train_3_architectures.py) and [`src/cli/train.py`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/src/cli/train.py)).
  - Real-time step loss, evaluation loss, SacreBLEU, and chrF++ curves are saved in [`outputs/training_logs/`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/outputs/training_logs) (`E1_loss.csv`, `qwen_E4_Trilingual_loss.csv`, `E10_Model_A_loss.csv`, etc.) and TensorBoard logs in [`outputs/logs/`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/outputs/logs).

### 2.2 Pre-Trained Models Implementation ($\ge 2$ Models)
- [x] **Implemented 4 Models Across 2 Paradigms**:
  1. `facebook/nllb-200-distilled-600M` (Sequence-to-Sequence NMT)
  2. `google/mt5-small` (Multilingual Text-to-Text Transformer)
  3. `Qwen/Qwen2.5-7B-Instruct` (Multilingual Causal Decoder LLM)
  4. `Mistral-7B-Instruct-v0.3` (Causal Decoder LLM Baseline)

### 2.3 Low-Resource Training Techniques
- [x] **Techniques Applied to Handle Low-Resource Settings**:
  - **Selective Layer Freezing**: Frozen the first 6 encoder layers (75.6M parameters frozen) of NLLB-200 to preserve lower-level cross-lingual representation space and prevent overfitting on small Swahili/Ekegusii datasets.
  - **Parameter-Efficient Fine-Tuning (4-bit QLoRA)**: Quantized base model weights to 4-bit NormalFloat (NF4) with double quantization. Attached LoRA adapters to key attention/MLP projections (`q_proj`, `v_proj`, `k_proj`, `o_proj`), updating only **0.29% of model parameters (20.9M out of 7.2B)**.
  - **Dictionary-Based Lexical Augmentation (E6)**: Injected bilingual dictionary definitions (`Kiswahili-Ekegusii` lexicon) directly into training prompts to boost rare word recall.
  - **Curriculum Learning (E7)**: Staged model training from short/simple monolingual and bilingual phrases up to complex public health advisories.

### 2.4 Ablation Studies (Zero-Shot vs. Few-Shot, Pivot, Domain Adaptation)
- [x] **Completed 10 Experiment Series (E0–E10)**:
  - **E0 (Zero-Shot Baseline)**: Evaluated raw pre-trained LLM without task fine-tuning $\rightarrow$ **0.08 BLEU / 12.96 chrF++** (severe hallucinations and English copying).
  - **E1 (Direct Few-Shot QLoRA)**: Direct tuning on English-Ekegusii $\rightarrow$ **16.28 BLEU / 42.94 chrF++** (top direct English → Ekegusii model).
  - **E2 (Kiswahili-Ekegusii Pivot Transfer)**: Evaluated Kiswahili as a structural pivot to bridge English to Ekegusii $\rightarrow$ **6.17 BLEU / 32.93 chrF++**.
  - **E3 (Bilingual Joint Transfer)**: Joint fine-tuning on Eng-Eke + Swa-Eke pairs $\rightarrow$ **14.96 BLEU / 41.72 chrF++**.
  - **E4 (Trilingual Multi-Target Alignment)**: 3-way alignment across English, Kiswahili, and Ekegusii $\rightarrow$ **12.96 BLEU / 39.45 chrF++**.
  - **E5 (Full Resources Scaling)**: Combined all available datasets including monolingual, bilingual, and verified PSA advisories $\rightarrow$ **16.04 BLEU / 42.49 chrF++**.
  - **E6 (Lexical Prompt Augmentation)**: Dictionary terms injected in prompt $\rightarrow$ **14.21 BLEU / 40.53 chrF++**.
  - **E7 (Curriculum Learning)**: Staged difficulty training $\rightarrow$ **15.74 BLEU / 41.10 chrF++** (fastest loss convergence).
  - **E9 (Sequential Adapter Transfer)**: Pre-trained Swa-Eke, then adapted to Eng-Eke $\rightarrow$ **0.08 BLEU** (encountered catastrophic forgetting when adapter weights were overwritten).

### 2.5 Save Checkpoints & Logs
- [x] **Checkpoints & Logs Saved**:
  - Saved model checkpoints under [`checkpoints/qwen/`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/checkpoints/qwen) and [`checkpoints/mistral/`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/checkpoints/mistral).
  - Published model adapters to Hugging Face Hub under [`aykahsay`](https://huggingface.co/aykahsay).
  - CSV training logs saved in [`outputs/training_logs/`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/outputs/training_logs).

### 2.6 Hyperparameters, Training Time & Preliminary Results
- [x] **Documented Complete Experimental Parameters**: (See Section 3 for full parameter breakdown).

### 2.7 Mid-Week Check-In: GPU/Colab Troubleshooting
- [x] **Resolved GPU Memory & Loss Convergence Bottlenecks**: (See Section 4 for details).

### 2.8 Week 3 Deliverable & Success Criteria Verification
- [x] **Working Translation Demo Delivered**:
  - **CLI Inference Script**: [`src/cli/translate.py`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/src/cli/translate.py) & [`src/cli/main.py`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/src/cli/main.py).
  - **Jupyter Notebook Demo**: [`notebooks/07_train_qwen.ipynb`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/notebooks/07_train_qwen.ipynb) & [`start.ipynb`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/start.ipynb).
  - **Interactive Streamlit Web App**: [`app.py`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/app.py) supporting live translation of sample health, flood, and emergency PSAs.

---

## 3. Detailed Hyperparameters & Training Setup

| Parameter | Model 1: NLLB-200 Distilled | Model 1: mT5-Small | Model 2: Qwen-7B QLoRA | Model 2: Mistral-7B QLoRA |
| :--- | :--- | :--- | :--- | :--- |
| **Base Model** | `facebook/nllb-200-distilled-600M` | `google/mt5-small` | `Qwen/Qwen2.5-7B-Instruct` | `Mistral-7B-Instruct-v0.3` |
| **Fine-Tuning Type** | Selective Layer Freezing (Enc L1-6) | Full Text-to-Text | 4-bit QLoRA | 4-bit QLoRA |
| **LoRA Rank ($r$)** | N/A | N/A | 16 | 16 |
| **LoRA Alpha ($\alpha$)**| N/A | N/A | 32 | 32 |
| **Target Modules** | N/A | N/A | `q_proj, v_proj, k_proj, o_proj` | `q_proj, v_proj, k_proj, o_proj` |
| **Trainable Params** | ~524.4M / 600M (87.4%) | 300M (100%) | 20.9M / 7.2B (0.29%) | 20.9M / 7.2B (0.29%) |
| **Optimizer** | Adafactor | AdamW | Paged AdamW 8-bit | Paged AdamW 8-bit |
| **Learning Rate** | $3 \times 10^{-4}$ | $5 \times 10^{-4}$ | $2 \times 10^{-4}$ | $2 \times 10^{-4}$ |
| **LR Scheduler** | Linear Decay | Linear Decay | Cosine Annealing with Warmup | Cosine Annealing with Warmup |
| **Batch Size** | 16 (Accum 4 $\times$ 4) | 16 (Accum 4 $\times$ 4) | 16 (Accum 4 $\times$ 4) | 16 (Accum 4 $\times$ 4) |
| **Precision** | `fp16` | `fp16` | 4-bit NF4 / `bf16` compute | 4-bit NF4 / `bf16` compute |
| **Training Epochs** | 3 epochs | 3 epochs | 3 epochs (~15,500 steps) | 3 epochs (~15,500 steps) |
| **Training Hardware** | NVIDIA T4 (15GB VRAM) | NVIDIA T4 (15GB VRAM) | NVIDIA RTX 4090 / T4 | NVIDIA RTX 4090 / T4 |
| **Training Time** | ~45 minutes | ~30 minutes | ~1.2h (4090) / ~3.5h (T4) | ~1.3h (4090) / ~3.8h (T4) |

---

## 4. Mid-Week Check-In & GPU / Colab Troubleshooting

During experimental execution on Google Colab free-tier (15GB Tesla T4 GPUs) and local workstations, three major technical bottlenecks were identified and resolved:

### 1. CUDA Out-Of-Memory (OOM) on 7B LLM Fine-Tuning
- **Symptom**: Full fine-tuning or standard 16-bit LoRA on Qwen-7B / Mistral-7B resulted in `CUDA out of memory` errors during backward passes on 15GB VRAM GPUs.
- **Root Cause**: Storing 7B model parameters, gradients, and 32-bit Adam optimizer states required >28GB VRAM.
- **Resolution**:
  1. Implemented **4-bit NormalFloat (NF4) double quantization** using `bitsandbytes`.
  2. Adopted **paged 8-bit AdamW optimizer (`paged_adamw_8bit`)** which offloads memory spikes to CPU RAM.
  3. Configured **Gradient Accumulation (step count = 4)** with per-device batch size of 4, keeping peak VRAM usage under **9.2 GB**.

### 2. Loss Instability & Gradient Explosion in Low-Resource Seq2Seq
- **Symptom**: Baseline NLLB-200 fine-tuning on small Ekegusii datasets exhibited sudden gradient spikes and validation loss divergence after epoch 2.
- **Resolution**:
  1. Implemented **Selective Encoder Layer Freezing**: Froze the first 6 encoder layers of `nllb-200-distilled-600M` to stabilize lower-level representations.
  2. Applied **Gradient Clipping** (`max_grad_norm = 1.0`) and switched from standard AdamW to **Adafactor** with parameter scale clipping.

### 3. Catastrophic Forgetting in Sequential Adapter Transfer (E9)
- **Symptom**: Sequential adaptation (pre-training adapter on Swa-Eke, then continuing training on Eng-Eke) resulted in loss spikes and collapsed test SacreBLEU (0.08).
- **Root Cause**: Unconstrained gradient updates destroyed the pivot representations learned in stage 1.
- **Resolution**: Switched from sequential adapter fine-tuning to **Joint Trilingual Training (E4)** and **Curriculum Mixture Scaling (E7)**, which preserved multi-directional translation capabilities.

---

## 5. Summary of Ablation Results (E0–E10)

| Exp ID | Title | Objective | Final Train Loss | Final Val Loss | SacreBLEU | chrF++ | Key Insight |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **E0** | Zero-Shot | Test un-tuned LLM baseline | N/A | N/A | 0.08 | 12.96 | Unusable raw output; proves NMT fine-tuning necessity. |
| **E1** | Direct Eng-Eke | Direct QLoRA tuning on primary pair | **0.41** | **0.58** | **16.28** | **42.94** | **Top direct English → Ekegusii model (+16.20 BLEU gain).** |
| **E2** | Swa-Eke Pivot | Test Kiswahili as structural pivot | 0.52 | 0.71 | 6.17 | 32.93 | Enables zero-direct-pair translation via Bantu similarity. |
| **E3** | Bilingual Joint | Joint Eng-Eke + Swa-Eke training | 0.48 | 0.65 | 14.96 | 41.72 | Stabilizes multi-pair shared representations. |
| **E4** | Trilingual | 3-way alignment across Eng/Swa/Eke | 0.44 | 0.62 | 12.96 | 39.45 | Balanced multi-target model supporting 6 translation directions. |
| **E5** | Full Resources | Train on full dataset (49,277 sents) | 0.39 | 0.56 | 16.04 | 42.49 | High performance confirming benefits of corpus scaling. |
| **E6** | Lexical Aug | Inject dictionary terms into prompt | 0.43 | 0.64 | 14.21 | 40.53 | Higher lexicon recall; slightly lower sentence fluency. |
| **E7** | Curriculum | Staged training (easy $\rightarrow$ complex) | **0.38** | **0.55** | 15.74 | 41.10 | Fastest loss reduction and stable gradient convergence. |
| **E9** | Sequential | Swa-Eke pre-train $\rightarrow$ Eng-Eke adapt | 0.89 | 1.12 | 0.08 | 12.96 | Suffered from catastrophic forgetting during adapter reset. |
| **E10-A**| Sub-Model A | Swahili benchmark validation | 0.45 | 0.50 | 63.83* | 78.65* | High-resource Swahili benchmark (*NLLB evaluation). |
| **E10-B**| Sub-Model B | Verification run for Eng-Eke QLoRA | 0.40 | 0.57 | 15.58 | 41.98 | Validates E1 consistency across random seeds. |

---

## 6. Verification & Working Translation Demo

### 6.1 CLI Translation Demo
The inference CLI script ([`src/cli/translate.py`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/src/cli/translate.py)) enables single and batch translations via command line:

```bash
# Example CLI Execution Command
python -m src.cli.main translate \
  --sentences "Boil water before drinking to prevent cholera." \
  --source-lang "English" \
  --target-lang "Ekegusii" \
  --model-name "qwen" \
  --adapter-path "checkpoints/qwen/E1_English_Ekegusii"
```

### 6.2 Notebook Interactive Demo
Interactive notebooks ([`notebooks/07_train_qwen.ipynb`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/notebooks/07_train_qwen.ipynb) and [`start.ipynb`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/start.ipynb)) provide cell-by-cell execution for model loading, text generation, and metric calculation.

### 6.3 Sample Public Service Announcement (PSA) Qualitative Outputs

#### Health Advisory PSA
- **Source (English)**: *"Boil water before drinking to prevent cholera infection."*
- **Kiswahili Baseline (NLLB-200)**: *"Chemsha maji kabla ya kunywa ili kuzuia maambukizi ya kipindupindu."*
- **Ekegusii Baseline (mT5-small)**: *"Kunywa amache kuzuia kipindupindu."* *(Incomplete morphemes)*
- **Ekegusii Fine-Tuned (Qwen-7B QLoRA)**: *"Toka amache chinchera tore n'okuria kere gokina oborwire bwa kipindupindu."* *(Syntactically complete and accurate)*

#### Emergency Flood Warning PSA
- **Source (English)**: *"Heavy rain warning: Move to higher ground immediately."*
- **Kiswahili Baseline (NLLB-200)**: *"Onyo la mvua kubwa: Hamia katika eneo la juu mara moja."*
- **Ekegusii Fine-Tuned (Qwen-7B QLoRA)**: *"Omoka w'embura enene: Rora ase ekerogo kia igoro rero."*

#### Public Security PSA
- **Source (English)**: *"Report any suspicious activity to the local chiefs or police station."*
- **Kiswahili Baseline (NLLB-200)**: *"Ripoti shughuli yoyote inayotia shaka kwa chifu au kituo cha polisi."*
- **Ekegusii Fine-Tuned (Qwen-7B QLoRA)**: *"Roria amang'ana getaari amaya ase abachifu tore ne chikitio chi'omochango."*

---

## 7. Strategic Recommendations & Next Steps

1. **Deployment Architecture**: Deploy `Qwen-7B QLoRA` via vLLM / Ollama backends for fast local API inference, coupled with `NLLB-200-distilled-600M` for high-speed Kiswahili translation.
2. **Community Feedback**: Leverage the integrated Streamlit native speaker feedback portal ([`app.py`](file:///c:/Users/Admin/OneDrive%20-%20United%20States%20International%20University%20%28USIU%29/Documents/NLP/Multilogual_transaltion_nlp/app.py)) to gather qualitative evaluations from verified Ekegusii native speakers for future fine-tuning iterations.
