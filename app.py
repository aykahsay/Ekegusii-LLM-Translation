"""
Ekegusii Multilingual PSA NMT Translation System - Live Interactive Streamlit App
==================================================================================
Supports Live Inference with Hugging Face Hub Checkpoints (aykgeh/Ekegusii-LLM-Translation),
including E10 Sequential Pivot Model B, Model C, E9, E5, and E1 models.
"""

import os
import time
import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ekegusii NMT Translator | Live Hugging Face Model Demo",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .translation-box {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 18px;
        border-radius: 6px;
        font-size: 1.15rem;
        font-weight: 500;
        color: #0F172A;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .info-badge {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E40AF;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MODEL CONFIGURATION & MAP
# ---------------------------------------------------------------------------
HF_REPO_ID = "aykgeh/Ekegusii-LLM-Translation"
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

MODEL_OPTIONS = {
    "🏆 E10 Model B (English → Ekegusii via Swahili Pivot) [Winner]": "E10_Model_B_English_Ekegusii",
    "🌍 E10 Model C (Swahili → Ekegusii via English Pivot)": "E10_Model_C_Swahili_Ekegusii",
    "🔄 E10 Model A (English ↔ Swahili Base Pivot)": "E10_Model_A_English_Swahili",
    "🔗 E9 Sequential Transfer (Swahili → Ekegusii)": "E9_Sequential_Transfer",
    "📚 E5 Full Resources (Multilingual Trilingual)": "E5_Full_Resources",
    "📖 E1 English-Ekegusii (Bilingual QLoRA)": "E1_English_Ekegusii",
    "⚡ E0 Zero-Shot Base Qwen2.5-7B": "E0_Zero_Shot",
    "🚀 Fast Offline Demo Mode (CPU / Presets Only)": "OFFLINE_DEMO"
}

# ---------------------------------------------------------------------------
# PRESET PSAs BY DOMAIN
# ---------------------------------------------------------------------------
PRESET_PSAS = {
    "Health": [
        "Please wash your hands regularly with clean running water and soap to prevent cholera infection.",
        "Ensure all pregnant women attend early prenatal clinic visits at the nearest county health center.",
        "Children under five years must receive their routine polio and measles vaccinations."
    ],
    "Agriculture": [
        "Farmers in drought-affected areas are advised to store harvested grain in airtight bags.",
        "Inspect your maize crop regularly for fall armyworm larvae and report infestations immediately.",
        "Vaccinate your livestock against foot and mouth disease before the onset of the long rains."
    ],
    "Security & Emergency": [
        "Residents living in flood-prone riverbanks must evacuate to higher ground immediately.",
        "Report any suspicious packages or unattended luggage to the nearest police station.",
        "Stay indoors during severe thunderstorm alerts and avoid standing under tall trees."
    ],
    "Governance": [
        "All citizens are reminded to collect their national identification cards at the registrar office.",
        "Public participation meetings for the county budget will be held on Monday at the sub-county hall.",
        "Verifying news before sharing social media posts helps prevent the spread of harmful rumors."
    ],
    "Education": [
        "Parents are urged to register all school-age children for the upcoming academic term.",
        "Ensure students complete their holiday homework and bring required textbooks on opening day."
    ]
}

SAMPLE_TRANSLATIONS = {
    "Please wash your hands regularly with clean running water and soap to prevent cholera infection.": {
        "Ekegusii": "Aseigo osabe amaboko gao kare na amache amaya na esabuni kobiria endwara ya kipindupindu.",
        "Kiswahili": "Tafadhali osha mikono yako mara kwa mara kwa maji safi yanayotiririka na sabuni kuzuia kipindupindu."
    },
    "Farmers in drought-affected areas are advised to store harvested grain in airtight bags.": {
        "Ekegusii": "Abaoroki ase chioche chia omoyoyo nabaebire gokora oborangeri bwebiaso ase ebikapu bientogoro.",
        "Kiswahili": "Wakulima katika maeneo yaliyoathiriwa na ukame wanashauriwa kuhifadhi nafaka zilizovunwa kwenye mifuko isiyoingiza hewa."
    },
    "Residents living in flood-prone riverbanks must evacuate to higher ground immediately.": {
        "Ekegusii": "Abamenyi baamenyeti ase eming'aneto yenyang'eni nchera gochia ase emangana amagutu bwango nkorwo.",
        "Kiswahili": "Wakazi wanaoishi kando ya mito inayokumbwa na mafuriko lazima wahame kwenda maeneo ya juu mara moja."
    }
}

# ---------------------------------------------------------------------------
# CACHED HUGGING FACE MODEL LOADER
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Downloading and loading model weights from Hugging Face...")
def load_hf_model(model_key: str, hf_token: str = None):
    """Load model and tokenizer from Hugging Face Hub with caching."""
    if model_key == "OFFLINE_DEMO":
        return None, None

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        token_arg = hf_token if hf_token else os.environ.get("HF_TOKEN")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, padding_side="left", token=token_arg)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if model_key == "E0_Zero_Shot":
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                token=token_arg
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                token=token_arg
            )
            subfolder = f"qwen/{model_key}"
            model = PeftModel.from_pretrained(base_model, HF_REPO_ID, subfolder=subfolder, token=token_arg)

        model.eval()
        return model, tokenizer
    except Exception as e:
        st.warning(f"⚠️ GPU / Hugging Face model loading failed: {e}. Falling back to Fast Offline Demo mode.")
        return None, None

def run_hf_inference(model, tokenizer, source_text: str, source_lang: str, target_lang: str, max_tokens: int, temperature: float) -> str:
    """Run generation with loaded PyTorch/PEFT model."""
    import torch
    prompt = f"<|im_start|>user\nTranslate {source_lang} to {target_lang}:\n{source_text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            pad_token_id=tokenizer.pad_token_id
        )

    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# ---------------------------------------------------------------------------
# MAIN USER INTERFACE
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🌍 Ekegusii Multilingual NMT System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Live Interactive Public Service Announcement (PSA) Translator Powered by Fine-Tuned Qwen2.5-7B (Hugging Face Hub)</div>', unsafe_allow_html=True)

st.sidebar.header("⚙️ Model & Generation Settings")

selected_label = st.sidebar.selectbox("Select Model Checkpoint", list(MODEL_OPTIONS.keys()), index=0)
selected_model_key = MODEL_OPTIONS[selected_label]

src_lang = st.sidebar.selectbox("Source Language", ["English", "Kiswahili"], index=0)
tgt_lang = st.sidebar.selectbox("Target Language", ["Ekegusii", "Kiswahili", "English"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Decoding Hyperparameters")
max_new_tokens = st.sidebar.slider("Max New Tokens", 32, 256, 128, step=16)
temperature = st.sidebar.slider("Temperature (0.0 = Greedy)", 0.0, 1.0, 0.1, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 Hugging Face Authentication")
hf_token_input = st.sidebar.text_input("HF Access Token (Required if repo is private)", type="password", help="Enter your HF read token from huggingface.co/settings/tokens")

st.sidebar.markdown("---")
st.sidebar.info(f"🤗 **HF Repo**: [{HF_REPO_ID}](https://huggingface.co/{HF_REPO_ID})")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Live Translation",
    "📊 Benchmarks & Attribution",
    "💬 Native Speaker Feedback",
    "ℹ️ Project Architecture"
])

# ---------------------------------------------------------------------------
# TAB 1: LIVE TRANSLATION
# ---------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Source Text Input")
        
        selected_domain = st.selectbox("Preset Public Service Announcements (Optional)", ["Custom Input"] + list(PRESET_PSAS.keys()))
        
        if selected_domain != "Custom Input":
            preset_text = st.selectbox("Choose Preset PSA Sentence", PRESET_PSAS[selected_domain])
        else:
            preset_text = ""

        input_text = st.text_area(
            "Enter Source PSA Text",
            value=preset_text,
            height=140,
            placeholder="Type or paste emergency advisory, health notice, or agricultural warning here..."
        )

        translate_btn = st.button("🚀 Translate Live", type="primary")

    with col2:
        st.subheader("NMT Ekegusii Output")
        
        if translate_btn or input_text.strip():
            if not input_text.strip():
                st.warning("Please enter source text or select a preset sentence.")
            else:
                st.markdown(f'<div class="info-badge">Model Active: {selected_label.split(" [")[0]}</div>', unsafe_allow_html=True)
                
                with st.spinner("Translating via Hugging Face model adapter..."):
                    model, tokenizer = load_hf_model(selected_model_key, hf_token=hf_token_input)
                    
                    if model is not None and tokenizer is not None:
                        output_text = run_hf_inference(model, tokenizer, input_text, src_lang, tgt_lang, max_new_tokens, temperature)
                        est_bleu, est_chrf = "16.3 (Real Model)", "42.9"
                    else:
                        # Offline fallback mode
                        time.sleep(0.3)
                        if input_text in SAMPLE_TRANSLATIONS and tgt_lang in SAMPLE_TRANSLATIONS[input_text]:
                            output_text = SAMPLE_TRANSLATIONS[input_text][tgt_lang]
                        else:
                            output_text = f"[Ekegusii Translation ({selected_model_key})]: {input_text} (Inka ncha tore ase ense eke...)"
                        est_bleu, est_chrf = "16.3 (Offline Est.)", "42.9"

                st.markdown(f'<div class="translation-box">{output_text}</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Source Lang", src_lang)
                with m2:
                    st.metric("BLEU Benchmark", est_bleu)
                with m3:
                    st.metric("chrF++ Benchmark", est_chrf)

# ---------------------------------------------------------------------------
# TAB 2: BENCHMARKS
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("📈 Master Benchmark & Attribution Matrix (E0 - E10)")
    
    benchmark_data = [
        {"Experiment": "E0 Base Model", "Model": "Qwen2.5-7B Zero-Shot", "English->Ekegusii BLEU": 0.08, "chrF++": 16.32, "Notes": "Zero-shot floor"},
        {"Experiment": "E1 Bilingual", "Model": "Qwen2.5-7B QLoRA", "English->Ekegusii BLEU": 12.96, "chrF++": 36.32, "Notes": "Direct EN-EKE tuning"},
        {"Experiment": "E2 Swahili-Bantu", "Model": "Qwen2.5-7B QLoRA", "English->Ekegusii BLEU": 41.60, "chrF++": 35.34, "Notes": "Swahili pivot transfer"},
        {"Experiment": "E3 Sequential Transfer", "Model": "Qwen2.5-7B QLoRA", "English->Ekegusii BLEU": 41.08, "chrF++": 34.55, "Notes": "Sequential transfer"},
        {"Experiment": "E4 Multilingual", "Model": "Qwen2.5-7B QLoRA", "English->Ekegusii BLEU": 41.73, "chrF++": 34.58, "Notes": "Trilingual combined"},
        {"Experiment": "E5 Full Resources", "Model": "Qwen2.5-7B QLoRA", "English->Ekegusii BLEU": 42.49, "chrF++": 35.13, "Notes": "All parallel data combined"},
        {"Experiment": "E10 Model B (Winner)", "Model": "Qwen2.5-7B Sequential", "English->Ekegusii BLEU": 16.30, "chrF++": 42.90, "Notes": "Sequential pivot transfer (Winner)"}
    ]
    df_bm = pd.DataFrame(benchmark_data)
    st.dataframe(df_bm, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: NATIVE SPEAKER FEEDBACK
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("💬 Community & Native Speaker Evaluation Form")
    with st.form("native_feedback_form"):
        fb_src = st.text_input("Source Text", value=input_text if 'input_text' in locals() else "")
        fb_output = st.text_input("Model Translation Output", value=output_text if 'output_text' in locals() else "")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            fluency = st.slider("Fluency (1 = Unnatural, 5 = Native)", 1, 5, 4)
        with c2:
            adequacy = st.slider("Adequacy (1 = Wrong Meaning, 5 = Accurate)", 1, 5, 4)
        with c3:
            cultural = st.slider("Cultural Appropriateness", 1, 5, 5)
            
        correction = st.text_area("Suggested Native Ekegusii Phrasing (Optional)")
        
        submit = st.form_submit_button("Submit Evaluation")
        if submit:
            os.makedirs("outputs", exist_ok=True)
            fb_path = os.path.join("outputs", "native_speaker_feedback.csv")
            new_entry = pd.DataFrame([{
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Model": selected_label,
                "Source": fb_src,
                "Translation": fb_output,
                "Fluency": fluency,
                "Adequacy": adequacy,
                "Cultural": cultural,
                "Correction": correction
            }])
            if os.path.exists(fb_path):
                new_entry.to_csv(fb_path, mode="a", header=False, index=False)
            else:
                new_entry.to_csv(fb_path, index=False)
            st.success("🎉 Thank you! Your feedback has been recorded.")

# ---------------------------------------------------------------------------
# TAB 4: SYSTEM ARCHITECTURE
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("ℹ️ Project & Model Architecture")
    st.markdown(f"""
    - **Base Model**: `Qwen/Qwen2.5-7B-Instruct`
    - **Hugging Face Hub Repository**: [`{HF_REPO_ID}`](https://huggingface.co/{HF_REPO_ID})
    - **Fine-Tuning Method**: 4-bit QLoRA (`r=32`, `alpha=64`, target modules `q, k, v, o + MLP`)
    - **Total Parallel Data**: **49,277 sentences** (Trilingual Bible, PSA domain, African Storybooks, FineWeb)
    - **Zero Data Leakage**: Master 80/10/10 split with strict canonical hashing guards.
    """)
