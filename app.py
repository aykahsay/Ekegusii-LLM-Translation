import streamlit as st
import pandas as pd
import numpy as np
import os, json, pickle
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="📢 Kenya Public Service Announcement (PSA) Translation Studio",
    page_icon="📢",
    layout="wide"
)

# --- LOAD CLASSIFIER MODEL ---
@st.cache_resource
def load_classifier():
    clf_path = "psa_classifier.pkl"
    vec_path = "tfidf_vectorizer.pkl"
    if os.path.exists(clf_path) and os.path.exists(vec_path):
        with open(clf_path, "rb") as f:
            clf = pickle.load(f)
        with open(vec_path, "rb") as f:
            vec = pickle.load(f)
        return clf, vec
    return None, None

# --- LOAD TRANSLATION PIPELINES ---
@st.cache_resource
def load_translation_pipeline(source_lang, target_lang):
    if source_lang == "English" and target_lang == "Kiswahili":
        model_path = "models/psa-en-sw-finetuned" if os.path.exists("models/psa-en-sw-finetuned") else "Helsinki-NLP/opus-mt-en-sw"
    elif source_lang == "Kiswahili" and target_lang == "English":
        model_path = "Helsinki-NLP/opus-mt-sw-en"
    elif source_lang == "English" and target_lang == "Ekegusii":
        model_path = "models/nllb-en-guz" if os.path.exists("models/nllb-en-guz") else "facebook/nllb-200-distilled-600M"
    else:
        return None
            
    try:
        if "nllb" in model_path:
            translator = pipeline("translation", model=model_path, tokenizer=model_path, src_lang="eng_Latn", tgt_lang="guz_Latn")
        else:
            translator = pipeline("translation", model=model_path, tokenizer=model_path)
        return translator
    except Exception:
        return None

# --- LOAD MASTER DATASET ---
@st.cache_data
def load_master_dataset():
    path = os.path.join("data", "Master_PSA_Only.csv")
    if os.path.exists(path):
        return pd.read_csv(path, dtype=str)
    return pd.DataFrame()

# --- PRESET EXAMPLE PSAs ---
PRESETS = {
    "Health": "Wash your hands frequently with soap and running water for 20 seconds to prevent the spread of diseases.",
    "Security": "Report suspicious activities and security threats in public places immediately to the nearest police station or hotline.",
    "Education": "Parents are encouraged to ensure all children attend school regularly and register for digital learning resources.",
    "Governance": "Citizens are invited to participate in the public budget hearings and submit comments to county representatives.",
    "Agriculture": "Adopt drought-resistant crops and sustainable water-harvesting techniques to protect your farm harvest."
}

def main():
    st.title("📢 Kenya Multilingual PSA Translation & Evaluation Studio")
    st.write("A specialized NLP platform for translating and evaluating Public Service Announcements in **Kiswahili** and **Ekegusii (Gusii)**.")

    clf, vec = load_classifier()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Translation Studio", 
        "🔍 PSA Verification Classifier", 
        "📚 Trilingual Parallel Corpus", 
        "⭐ Human Evaluation Form"
    ])

    # --- TAB 1: TRANSLATION STUDIO ---
    with tab1:
        st.header("Interactive Multilingual PSA Translator")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Settings & Presets")
            target_lang = st.selectbox("Select Target Language:", ["Kiswahili", "Ekegusii"])
            domain_preset = st.selectbox("Load Example PSA Preset:", ["Custom Input"] + list(PRESETS.keys()))
            
            initial_text = ""
            if domain_preset != "Custom Input":
                initial_text = PRESETS[domain_preset]
                
            st.info(f"Targeting: **{target_lang}**")
            
        with col2:
            input_text = st.text_area("Enter English Public Service Announcement (PSA):", value=initial_text, height=140,
                                      placeholder="Type or paste your public advisory sentence here...")
            
            if st.button("Translate Advisory 🔄", type="primary"):
                if not input_text.strip():
                    st.warning("Please enter text to translate.")
                else:
                    translator = load_translation_pipeline("English", target_lang)
                    
                    if translator is None:
                        st.info(f"Using fallback demonstration pipeline for {target_lang}...")
                        translated_text = f"[Demonstration Translation to {target_lang}]: {input_text}"
                    else:
                        with st.spinner(f"Translating to {target_lang}..."):
                            res = translator(input_text)[0]
                            translated_text = res.get('translation_text', str(res))

                    st.success(f"Translation Complete ({target_lang})")
                    st.text_area(f"{target_lang} Output:", value=translated_text, height=130)

                    # Compute Classifier Score if available
                    if clf and vec:
                        X = vec.transform([input_text])
                        prob = clf.predict_proba(X)[0][1]
                        is_psa = prob >= 0.60
                        st.metric("PSA Confidence Score", f"{prob:.1%}", delta="Valid PSA" if is_psa else "Non-PSA")

    # --- TAB 2: PSA VERIFICATION CLASSIFIER ---
    with tab2:
        st.header("Public Service Announcement Classifier")
        st.write("Verifies whether an input sentence qualifies as a valid Public Service Announcement using TF-IDF + Logistic Regression.")
        
        check_text = st.text_area("Enter Sentence to Classify:", height=100, 
                                  placeholder="e.g. Wash hands to stop disease spread.")
        if st.button("Classify Sentence 🔍"):
            if not check_text.strip():
                st.warning("Please enter a sentence.")
            elif clf and vec:
                X = vec.transform([check_text])
                prob = clf.predict_proba(X)[0][1]
                is_psa = prob >= 0.60
                
                if is_psa:
                    st.success(f"✅ Verdict: Confirmed Public Service Announcement (Confidence: {prob:.1%})")
                else:
                    st.error(f"❌ Verdict: General Text / Non-PSA (Confidence: {prob:.1%})")
            else:
                st.error("Classifier model files not found in root directory.")

    # --- TAB 3: TRILINGUAL PARALLEL CORPUS ---
    with tab3:
        st.header("Trilingual Parallel Dataset Browser")
        df_master = load_master_dataset()
        if not df_master.empty:
            domains = ["All"] + list(df_master['Domain'].dropna().unique())
            selected_domain = st.selectbox("Filter Corpus by Domain:", domains)
            
            if selected_domain != "All":
                filtered_df = df_master[df_master['Domain'] == selected_domain]
            else:
                filtered_df = df_master
                
            st.dataframe(filtered_df, use_container_width=True)
            st.caption(f"Displaying {len(filtered_df):,} entries.")

    # --- TAB 4: HUMAN EVALUATION FORM ---
    with tab4:
        st.header("Native Speaker Human Evaluation Form")
        st.write("Score translations on **Fluency**, **Adequacy**, and **Cultural Accuracy** (1 to 5 scale).")
        
        eval_en = st.text_input("Source English Text:", value="Wash your hands frequently with soap to prevent disease.")
        eval_sw = st.text_input("Swahili Translation:", value="Nawa mikono yako mara kwa mara kwa sabuni kuzuia magonjwa.")
        eval_guz = st.text_input("Ekegusii Translation:", value="Esibie amaboko ao botambe na esabuni gotanga amarwaire.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            fluency = st.slider("Fluency (1=Poor, 5=Native)", 1, 5, 4)
        with c2:
            adequacy = st.slider("Adequacy / Meaning (1=Low, 5=Exact)", 1, 5, 5)
        with c3:
            cultural = st.slider("Cultural Accuracy (1=Poor, 5=Natural)", 1, 5, 4)
            
        notes = st.text_area("Evaluator Feedback / Corrections:")
        
        if st.button("Submit Evaluation Score ⭐"):
            log_file = os.path.join("data", "human_eval_feedback_log.csv")
            new_entry = {
                "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "English": eval_en,
                "Kiswahili": eval_sw,
                "Ekegusii": eval_guz,
                "Fluency": fluency,
                "Adequacy": adequacy,
                "Cultural_Accuracy": cultural,
                "Notes": notes
            }
            log_df = pd.DataFrame([new_entry])
            if os.path.exists(log_file):
                log_df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                log_df.to_csv(log_file, index=False)
            st.success("Thank you! Human evaluation score saved successfully.")

if __name__ == "__main__":
    main()
