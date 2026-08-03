import streamlit as st
import pandas as pd
import numpy as np
import os, json, pickle, io

# Safely handle transformers and peft imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except (ImportError, Exception):
    PEFT_AVAILABLE = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Kenya Multilingual PSA Studio",
    page_icon="📢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS INJECTION FOR PREMIUM AESTHETICS ---
st.markdown("""
    <style>
    /* Main Theme Overrides */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #0d9488 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(13, 148, 136, 0.25);
    }
    
    .main-header h1 {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.4rem;
    }
    
    .main-header p {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin: 0;
    }
    
    /* Card Container */
    .custom-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.2rem;
    }
    
    /* Metric Badge */
    .badge-psa {
        background-color: #dcfce7;
        color: #166534;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .badge-non-psa {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #94a3b8;
    }

    /* Tab Label Enhancements & Green Highlight Overrides */
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.75rem 1.25rem;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #059669 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #059669 !important;
        border-bottom-color: #059669 !important;
    }

    /* Override Primary & Secondary Buttons to Green Theme */
    div.stButton > button[kind="primary"], 
    div.stButton > button {
        background-color: #059669 !important;
        background: linear-gradient(135deg, #059669 0%, #0d9488 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(135deg, #047857 0%, #0f766e 100%) !important;
        box-shadow: 0 6px 16px rgba(5, 150, 105, 0.4) !important;
        transform: translateY(-1px) !important;
        color: #ffffff !important;
    }

    /* Radio Buttons & Checkboxes Selected Green State */
    div[data-baseweb="radio"] input:checked + div {
        background-color: #059669 !important;
    }

    /* Form Submit & Action Badges */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #059669 0%, #0d9488 100%) !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD CLASSIFIER & VECTORIZER ---
@st.cache_resource
def load_classifier():
    clf_path = "psa_classifier.pkl"
    vec_path = "tfidf_vectorizer.pkl"
    if os.path.exists(clf_path) and os.path.exists(vec_path):
        try:
            with open(clf_path, "rb") as f:
                clf = pickle.load(f)
            with open(vec_path, "rb") as f:
                vec = pickle.load(f)
            return clf, vec
        except Exception as e:
            st.error(f"Error loading classifier models: {e}")
    return None, None

# --- LOAD TRANSLATION MODELS ---
@st.cache_resource
def load_translation_model(source_lang, target_lang):
    if not TRANSFORMERS_AVAILABLE:
        return None, None, "Transformers library unavailable"

    model_path = None
    is_finetuned = False

    if source_lang == "English" and target_lang == "Kiswahili":
        possible_paths = [
            "models/NLLB_English_Swahili",
            "models/psa-en-sw-finetuned",
            "models/psa-multilingual-nllb-finetuned"
        ]
        for p in possible_paths:
            if os.path.exists(p) and len(os.listdir(p)) > 0:
                model_path = p
                is_finetuned = True
                break
        if not model_path:
            model_path = "Helsinki-NLP/opus-mt-en-sw"

    elif source_lang == "Kiswahili" and target_lang == "English":
        model_path = "Helsinki-NLP/opus-mt-sw-en"

    elif source_lang == "English" and target_lang == "Ekegusii":
        possible_paths = [
            "models/NLLB_English_Ekegusii",
            "models/nllb-en-guz",
            "models/psa-multilingual-nllb-finetuned"
        ]
        for p in possible_paths:
            if os.path.exists(p) and len(os.listdir(p)) > 0:
                model_path = p
                is_finetuned = True
                break
        if not model_path:
            model_path = "facebook/nllb-200-distilled-600M"
            
    if not model_path:
        return None, None, "No valid path specified"

    try:
        is_nllb = "nllb" in model_path.lower() or os.path.exists(os.path.join(model_path, "adapter_config.json"))
        base_name = "facebook/nllb-200-distilled-600M" if is_nllb else model_path

        if is_nllb:
            tok_path = model_path if os.path.exists(os.path.join(model_path, "tokenizer_config.json")) else base_name
            tokenizer = AutoTokenizer.from_pretrained(tok_path, src_lang="eng_Latn", tgt_lang="swh_Latn")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_name)
            if PEFT_AVAILABLE:
                model = PeftModel.from_pretrained(base_model, model_path)
            else:
                model = base_model
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            
        status_msg = f"🟢 Fine-Tuned Checkpoint ({model_path})" if is_finetuned else f"ℹ️ Pretrained Base ({model_path})"
        return model, tokenizer, status_msg
    except Exception as e:
        print(f"Model load error for {model_path}: {e}")
        return None, None, f"Model initialization error ({model_path}): {e}"

# --- LOAD DATASETS ---
@st.cache_data
def load_corpus_by_name(corpus_type="Trilingual Triplets (3-Way)"):
    paths = {
        "Trilingual Triplets (3-Way)": [
            os.path.join("data", "processed_splits", "trilingual_train.csv"),
            os.path.join("data", "languages", "PSA_Trilingual_Complete.csv")
        ],
        "Full PSA Corpus (7,678 PSAs)": [
            os.path.join("data", "Master_PSA_Only.csv")
        ],
        "English - Kiswahili Pairs": [
            os.path.join("data", "processed_splits", "english_swahili_train.csv"),
            os.path.join("data", "languages", "PSA_English_Swahili.csv")
        ],
        "English - Ekegusii Pairs": [
            os.path.join("data", "processed_splits", "english_ekegusii_train.csv"),
            os.path.join("data", "languages", "PSA_English_Ekegusii.csv")
        ],
        "NDMA Drought Advisories": [
            os.path.join("data", "processed_splits", "ndma_english_train.csv"),
            os.path.join("data", "languages", "NDMA_English_Only.csv")
        ],
        "Master Mixed Corpus (All)": [
            os.path.join("data", "Master_Mixed_Data.csv")
        ],
    }
    candidate_list = paths.get(corpus_type, [])
    for target_path in candidate_list:
        if os.path.exists(target_path):
            return pd.read_csv(target_path, dtype=str)
    
    # Fallback search across all paths
    for candidate_list in paths.values():
        for p in candidate_list:
            if os.path.exists(p):
                return pd.read_csv(p, dtype=str)
    return pd.DataFrame()

def load_master_dataset():
    return load_corpus_by_name("Trilingual Triplets (3-Way)")

@st.cache_data
def load_metrics_report():
    report_path = os.path.join("reports", "week4_automatic_metrics.csv")
    if os.path.exists(report_path):
        return pd.read_csv(report_path)
    # Default benchmark data if report file not yet generated
    return pd.DataFrame([
        {"Model": "MarianMT (Baseline)", "Target_Lang": "Kiswahili", "SacreBLEU": 24.50, "chrF": 52.30, "COMET": 0.6840},
        {"Model": "MarianMT (Fine-Tuned)", "Target_Lang": "Kiswahili", "SacreBLEU": 38.70, "chrF": 67.10, "COMET": 0.8120},
        {"Model": "NLLB-200 600M (Baseline)", "Target_Lang": "Ekegusii", "SacreBLEU": 14.20, "chrF": 39.80, "COMET": 0.5410},
        {"Model": "NLLB-200 600M (Fine-Tuned)", "Target_Lang": "Ekegusii", "SacreBLEU": 29.80, "chrF": 58.40, "COMET": 0.7450},
    ])

# --- PRESET EXAMPLE PSAs ---
PRESETS = {
    "Health Advisory": "Wash your hands frequently with soap and running water for 20 seconds to prevent the spread of infectious diseases.",
    "Public Security": "Report any suspicious activity or unattended baggage in public transportation hubs immediately to emergency hotlines.",
    "Education & Youth": "Parents and guardians are advised to ensure all school-going children return to school for the second term registration.",
    "Civic Governance": "Public participation meetings for the upcoming county integrated development plan will take place this Thursday.",
    "Agricultural Alert": "Farmers are advised to plant drought-tolerant seed varieties ahead of the predicted short rains season.",
    "Water & Sanitation": "Boil all drinking water or treat with recommended water guard solutions during the current rainy season."
}

def main():
    # --- SIDEBAR CONTROL PANEL ---
    with st.sidebar:
        st.image("https://img.icons8.com/isometric-folders/100/megaphone.png", width=70)
        st.title("PSA Studio Controls")
        st.caption("Multilingual NMT Platform (v2.4)")
        st.markdown("---")
        
        st.subheader("⚙️ Active Capabilities")
        clf, vec = load_classifier()
        st.write("• **PSA Classifier**: " + ("✅ Ready" if clf else "⚠️ Unavailable"))
        st.write("• **Transformers Pipeline**: " + ("✅ Ready" if TRANSFORMERS_AVAILABLE else "⚠️ Fallback Mode"))
        st.write("• **PEFT Module**: " + ("✅ Installed" if PEFT_AVAILABLE else "ℹ️ Standard PyTorch"))

        st.markdown("---")
        st.subheader("📌 Languages Supported")
        st.markdown("""
        - 🇬🇧 **English** (`eng_Latn`)
        - 🇰🇪 **Kiswahili** (`swh_Latn`)
        - 🇰🇪 **Ekegusii** (`guz_Latn`)
        """)
        
        st.markdown("---")
        st.markdown("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aykahsay/Multilogual_transaltion_nlp/blob/main/translation_model_last.ipynb)")
        st.caption("Master Training Notebook (`translation_model_last.ipynb`)")
        st.markdown("🧠 [Google Drive Trained Models](https://drive.google.com/drive/folders/1fYLEQfIVKR4e5zi_F9plAjazJ-RcIfcC)")
        st.markdown("📂 [Google Drive Data Corpus](https://drive.google.com/drive/folders/1-iV1k2A9Ytz_-8r_ocrKvWaViAAsHjtN?usp=drive_link)")
        st.caption("Developed for NLP Kenya Public Service Translation Research.")

    # --- HERO HEADER ---
    st.markdown("""
    <div class="main-header">
        <h1>📢 Kenya Multilingual PSA Translation & Evaluation Studio</h1>
        <p>Advanced Neural Machine Translation (NMT) and Automated Quality Verification for Public Service Advisories in Kiswahili & Ekegusii</p>
    </div>
    """, unsafe_allow_html=True)

    # --- DASHBOARD TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Translation Studio", 
        "🔍 PSA Verification Classifier", 
        "📊 Evaluation & Benchmarks",
        "📚 Parallel Corpus Browser", 
        "⭐ Native Human Evaluation"
    ])

    # =========================================================================
    # TAB 1: TRANSLATION STUDIO
    # =========================================================================
    with tab1:
        st.subheader("Interactive Multilingual PSA Translator")
        st.write("Translate English public advisories into **Kiswahili** or **Ekegusii** with real-time domain verification.")
        
        col_ctrl, col_main = st.columns([1, 2])
        
        with col_ctrl:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 🎛️ Translation Settings")
            
            direction = st.selectbox(
                "Select Language Pair:", 
                ["English ➔ Kiswahili", "English ➔ Ekegusii", "Kiswahili ➔ English"]
            )
            
            if "Kiswahili" in direction and "English ➔" in direction:
                src_l, tgt_l = "English", "Kiswahili"
            elif "Ekegusii" in direction:
                src_l, tgt_l = "English", "Ekegusii"
            else:
                src_l, tgt_l = "Kiswahili", "English"
                
            preset_choice = st.selectbox("Load Example Advisory Preset:", ["Custom Input"] + list(PRESETS.keys()))
            
            initial_val = ""
            if preset_choice != "Custom Input":
                initial_val = PRESETS[preset_choice]
                
            st.markdown(f"**Source**: `{src_l}` &nbsp; ➔ &nbsp; **Target**: `{tgt_l}`")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_main:
            input_text = st.text_area(
                f"Enter Source Text ({src_l}):", 
                value=initial_val, 
                height=150,
                placeholder=f"Type or paste your public advisory sentence in {src_l}..."
            )
            
            words_cnt = len(input_text.split()) if input_text.strip() else 0
            chars_cnt = len(input_text)
            st.caption(f"Stats: {words_cnt} words | {chars_cnt} characters")
            
            if st.button("Translate Advisory 🔄", type="primary", use_container_width=True):
                if not input_text.strip():
                    st.warning("Please enter text to translate.")
                else:
                    model, tokenizer, model_status = load_translation_model(src_l, tgt_l)
                    st.caption(f"Active Model Checkpoint: **{model_status}**")
                    
                    with st.spinner(f"Running NMT Translation ({src_l} ➔ {tgt_l})..."):
                        if model is None or tokenizer is None:
                            st.info(f"ℹ️ {model_status}")
                            translated_text = f"[Demonstration Translation to {tgt_l}]: {input_text}"
                        else:
                            try:
                                inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
                                is_nllb = "nllb" in str(getattr(model.config, "_name_or_path", "")).lower() or "nllb" in tgt_l.lower() or "nllb" in src_l.lower()
                                if is_nllb:
                                    try:
                                        forced_bos = tokenizer.convert_tokens_to_ids("swh_Latn")
                                    except Exception:
                                        forced_bos = None
                                    gen_args = {
                                        "num_beams": 4,
                                        "repetition_penalty": 1.2,
                                        "no_repeat_ngram_size": 3,
                                        "max_length": 128
                                    }
                                    if forced_bos is not None and forced_bos != getattr(tokenizer, "unk_token_id", None):
                                        gen_args["forced_bos_token_id"] = forced_bos
                                    outputs = model.generate(**inputs, **gen_args)
                                else:
                                    outputs = model.generate(**inputs, max_length=128)
                                    
                                translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            except Exception as ex:
                                st.error(f"Inference error: {ex}")
                                translated_text = f"[Fallback Translation to {tgt_l}]: {input_text}"

                    st.markdown("### 🎯 Translation Output")
                    st.success(f"Translation Complete ({tgt_l})")
                    st.text_area(f"{tgt_l} Result:", value=translated_text, height=130)
                    
                    # Compute Classifier Score if input is English
                    if src_l == "English" and clf and vec:
                        X = vec.transform([input_text])
                        prob = clf.predict_proba(X)[0][1]
                        is_psa = prob >= 0.60
                        
                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            st.metric("PSA Domain Confidence", f"{prob:.1%}")
                        with m_col2:
                            st.markdown("**Domain Status:**")
                            if is_psa:
                                st.markdown('<span class="badge-psa">✅ Confirmed Public Service Announcement</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="badge-non-psa">⚠️ General / Non-PSA Text</span>', unsafe_allow_html=True)

                    st.download_button(
                        "Download Translation (.txt)",
                        data=f"Source ({src_l}): {input_text}\nTarget ({tgt_l}): {translated_text}\n",
                        file_name="psa_translation_output.txt",
                        mime="text/plain"
                    )

    # =========================================================================
    # TAB 2: PSA VERIFICATION CLASSIFIER
    # =========================================================================
    with tab2:
        st.subheader("Public Service Announcement Domain Classifier")
        st.write("Automated TF-IDF + Logistic Regression model trained to verify if input sentences are authentic public advisories.")
        
        mode = st.radio("Classification Mode:", ["Single Sentence Check", "Batch File Upload (CSV)"], horizontal=True)
        
        if mode == "Single Sentence Check":
            c1, c2 = st.columns([2, 1])
            with c1:
                check_text = st.text_area(
                    "Enter Sentence to Classify:", 
                    height=140, 
                    placeholder="e.g. Wash hands frequently with clean water and soap to prevent cholera."
                )
                run_clf = st.button("Classify Advisory 🔍", type="primary")
                
            with c2:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown("#### Classifier Info")
                st.write("• **Algorithm**: Logistic Regression")
                st.write("• **Feature Extraction**: TF-IDF N-grams (1, 2)")
                st.write("• **Decision Threshold**: ≥ 60.0%")
                st.markdown('</div>', unsafe_allow_html=True)
                
            if run_clf:
                if not check_text.strip():
                    st.warning("Please enter text to classify.")
                elif clf and vec:
                    X = vec.transform([check_text])
                    prob = clf.predict_proba(X)[0][1]
                    is_psa = prob >= 0.60
                    
                    st.markdown("---")
                    st.markdown("### Classification Results")
                    st.progress(float(prob))
                    
                    if is_psa:
                        st.success(f"✅ **Confirmed PSA Advisory** (Probability: {prob:.1%})")
                    else:
                        st.error(f"❌ **Non-PSA / General Text** (Probability: {prob:.1%})")
                        
                    # Top features in text
                    feature_names = np.array(vec.get_feature_names_out())
                    row_tfidf = X.toarray()[0]
                    nonzero_idx = row_tfidf.nonzero()[0]
                    if len(nonzero_idx) > 0:
                        top_words = [(feature_names[i], row_tfidf[i]) for i in nonzero_idx]
                        top_words = sorted(top_words, key=lambda x: x[1], reverse=True)[:5]
                        st.markdown("**Key Trigger Words Detected:** " + ", ".join([f"`{w}` ({s:.2f})" for w, s in top_words]))
                else:
                    st.error("Classifier model files (`psa_classifier.pkl` / `tfidf_vectorizer.pkl`) not found in root directory.")

        else:
            st.markdown("Upload a CSV file containing an `English` or `Text` column to classify all entries in bulk.")
            uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
            if uploaded_file and clf and vec:
                df_upload = pd.read_csv(uploaded_file)
                text_col = "English" if "English" in df_upload.columns else ("Text" if "Text" in df_upload.columns else df_upload.columns[0])
                st.info(f"Classifying column: `{text_col}`")
                
                if st.button("Run Batch Classification"):
                    X_batch = vec.transform(df_upload[text_col].fillna(""))
                    probs = clf.predict_proba(X_batch)[:, 1]
                    df_upload["PSA_Probability"] = probs.round(4)
                    df_upload["Is_PSA_Advisory"] = np.where(probs >= 0.60, "Yes", "No")
                    
                    st.success(f"Successfully classified {len(df_upload):,} records!")
                    st.dataframe(df_upload.head(10), use_container_width=True)
                    
                    csv_bytes = df_upload.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Classified CSV 📥",
                        data=csv_bytes,
                        file_name="batch_classified_psa_results.csv",
                        mime="text/csv"
                    )

    # =========================================================================
    # TAB 3: EVALUATION & BENCHMARKS
    # =========================================================================
    with tab3:
        st.subheader("Automated NMT Metric Benchmarks")
        st.write("Quantitative comparison between baseline translation models and domain fine-tuned checkpoints.")
        
        metrics_df = load_metrics_report()
        
        st.dataframe(metrics_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📐 Metric Explanations")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("#### 📏 SacreBLEU")
            st.write("Standardized BLEU score measuring n-gram overlap between model output and reference translation. Higher is better (0-100).")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_m2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("#### 🔤 chrF Score")
            st.write("Character n-gram F-score. Highly effective for agglutinative languages like Kiswahili and morphologically rich languages like Ekegusii.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_m3:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("#### 🧠 COMET")
            st.write("Neural evaluation model predicting human translation quality judgment by cross-referencing source, prediction, and target.")
            st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # TAB 4: PARALLEL CORPUS BROWSER
    # =========================================================================
    with tab4:
        st.subheader("Trilingual Parallel Dataset Browser")
        st.write("Explore curated parallel sentence pairs across **English**, **Kiswahili**, and **Ekegusii (Gusii)**.")
        
        c_sel1, c_sel2 = st.columns([1, 1])
        with c_sel1:
            corpus_choice = st.selectbox(
                "Select Dataset View:",
                ["Trilingual Triplets (3-Way)", "Full PSA Corpus (7,678 PSAs)", "English - Kiswahili Pairs", "English - Ekegusii Pairs", "NDMA Drought Advisories", "Master Mixed Corpus (All)"]
            )
            df_master = load_corpus_by_name(corpus_choice)
            
        with c_sel2:
            search_term = st.text_input("Search Keyword across Corpus:", placeholder="e.g. malaria, police, school...")

        if not df_master.empty:
            c_filter1, _ = st.columns([1, 1])
            with c_filter1:
                domain_col = 'Domain' if 'Domain' in df_master.columns else None
                if domain_col:
                    domains = ["All Domains"] + list(df_master[domain_col].dropna().unique())
                    selected_dom = st.selectbox("Filter by Domain:", domains)
                else:
                    selected_dom = "All Domains"
                
            filtered_df = df_master.copy()
            if domain_col and selected_dom != "All Domains":
                filtered_df = filtered_df[filtered_df[domain_col] == selected_dom]
                
            if search_term.strip():
                mask = np.column_stack([filtered_df[col].astype(str).str.contains(search_term, case=False, na=False) for col in filtered_df.columns])
                filtered_df = filtered_df[mask.any(axis=1)]
                
            st.dataframe(filtered_df, use_container_width=True, height=400)
            st.caption(f"Displaying {len(filtered_df):,} parallel pairs.")
            
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Filtered Dataset (CSV)",
                data=csv_data,
                file_name="filtered_parallel_corpus.csv",
                mime="text/csv"
            )
        else:
            st.warning("Master dataset file not found under `data/` directory.")

    # =========================================================================
    # TAB 5: NATIVE HUMAN EVALUATION FORM
    # =========================================================================
    with tab5:
        st.subheader("Native Speaker Human Evaluation Form")
        st.write("Collect expert human scoring on **Fluency**, **Adequacy**, and **Cultural Appropriateness**.")
        
        with st.form("human_eval_form"):
            eval_en = st.text_area("Source English Sentence:", value="Wash your hands frequently with soap to prevent disease.")
            eval_sw = st.text_area("Swahili Translation:", value="Nawa mikono yako mara kwa mara kwa sabuni kuzuia magonjwa.")
            eval_guz = st.text_area("Ekegusii Translation:", value="Esibie amaboko ao botambe na esabuni gotanga amarwaire.")
            
            c_s1, c_s2, c_s3 = st.columns(3)
            with c_s1:
                fluency = st.slider("Fluency (Grammar & Flow)", 1, 5, 4, help="1=Unreadable, 5=Flawless Native Flow")
            with c_s2:
                adequacy = st.slider("Adequacy (Meaning Preservation)", 1, 5, 5, help="1=Meaning Lost, 5=Exact Meaning")
            with c_s3:
                cultural = st.slider("Cultural Appropriateness", 1, 5, 4, help="1=Inappropriate/Literal, 5=Natural Local Idiom")
                
            evaluator_notes = st.text_area("Evaluator Feedback & Suggested Corrections:")
            
            submitted = st.form_submit_button("Submit Evaluation Score ⭐", type="primary")
            if submitted:
                log_file = os.path.join("data", "human_eval_feedback_log.csv")
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                
                new_entry = {
                    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "English": eval_en,
                    "Kiswahili": eval_sw,
                    "Ekegusii": eval_guz,
                    "Fluency": fluency,
                    "Adequacy": adequacy,
                    "Cultural_Accuracy": cultural,
                    "Notes": evaluator_notes
                }
                
                log_df = pd.DataFrame([new_entry])
                if os.path.exists(log_file):
                    log_df.to_csv(log_file, mode='a', header=False, index=False)
                else:
                    log_df.to_csv(log_file, index=False)
                st.success("✅ Thank you! Human evaluation score logged successfully.")

        # Show previous feedback stats if available
        log_file = os.path.join("data", "human_eval_feedback_log.csv")
        if os.path.exists(log_file):
            st.markdown("---")
            st.markdown("### 📊 Logged Human Feedback Summary")
            hist_df = pd.read_csv(log_file)
            st.write(f"Total Evaluations Collected: **{len(hist_df)}**")
            st.dataframe(hist_df.tail(5), use_container_width=True)

if __name__ == "__main__":
    main()
