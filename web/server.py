"""
Ekegusii Multilingual NMT - Cloudflare FastAPI Web Server
==========================================================
Serves static HTML/CSS/JS frontend from GitHub/Local workspace and connects to
Hugging Face model repository (aykgeh/Ekegusii-LLM-Translation).
Runs on port 8000 for Cloudflare Tunnel (cloudflared trycloudflare) deployment.
"""

import os
import sys
import time
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EkegusiiServer")

# Global PyTorch Model & Tokenizer cache
MODEL_CACHE = {
    "base_model": None,
    "tokenizer": None,
    "active_peft": None,
    "active_subfolder": "qwen/E1_English_Ekegusii/checkpoint-8000"
}

HF_REPO_ID = "aykgeh/Ekegusii-LLM-Translation"
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

app = FastAPI(title="Ekegusii NMT Cloudflare Server", version="1.0.0")

# Enable CORS for Cloudflare tunnel requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "English"
    target_lang: str = "Ekegusii"
    temperature: float = 0.1
    max_new_tokens: int = 128
    repetition_penalty: float = 1.1

class SwitchModelRequest(BaseModel):
    adapter_subfolder: str

CHECKPOINT_SUBMAP = {
    "E10_Model_B_English_Ekegusii": "qwen/E10_Model_B_English_Ekegusii/checkpoint-8000",
    "E10_Model_C_Swahili_Ekegusii": "qwen/E10_Model_C_Swahili_Ekegusii/checkpoint-6000",
    "E10_Model_A_English_Swahili": "qwen/E10_Model_A_English_Swahili/checkpoint-4500",
    "E9_Sequential_Transfer": "qwen/E9_Sequential_Transfer/checkpoint-8000",
    "E7_Curriculum_Learning": "qwen/E7_Curriculum_Learning/checkpoint-18000",
    "E6_Lexical_Augmentation": "qwen/E6_Lexical_Augmentation/checkpoint-7000",
    "E5_Full_Resources": "qwen/E5_Full_Resources/checkpoint-18000",
    "E4_Trilingual": "qwen/E4_Trilingual/checkpoint-15500",
    "E3_Bilingual": "qwen/E3_Bilingual/checkpoint-13000",
    "E2_Swahili_Ekegusii": "qwen/E2_Swahili_Ekegusii/checkpoint-6000",
    "E1_English_Ekegusii": "qwen/E1_English_Ekegusii/checkpoint-8000",
}

def resolve_subfolder(key: str) -> str:
    """Resolve model key or shorthand to exact HF Hub checkpoint path."""
    if key in CHECKPOINT_SUBMAP:
        return CHECKPOINT_SUBMAP[key]
    for exp, full_path in CHECKPOINT_SUBMAP.items():
        if exp in key:
            return full_path
    return key

# ---------------------------------------------------------------------------
# Model Loader Helper
# ---------------------------------------------------------------------------
def initialize_hf_model(subfolder: str = "qwen/E1_English_Ekegusii/checkpoint-8000", hf_token: Optional[str] = None):
    """Load Qwen base model in 4-bit and attach PEFT adapter from Hugging Face Hub."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    resolved_subfolder = resolve_subfolder(subfolder)
    token_val = hf_token or os.environ.get("HF_TOKEN") or None
    device_map = {"": 0} if torch.cuda.is_available() else "auto"

    logger.info(f"Loading Base Model: {BASE_MODEL_ID}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, padding_side="left", token=token_val)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        token=token_val
    )

    logger.info(f"Attaching LoRA Adapter from HF Repo '{HF_REPO_ID}', subfolder '{resolved_subfolder}'...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        HF_REPO_ID,
        subfolder=resolved_subfolder,
        token=token_val
    )
    peft_model.eval()

    MODEL_CACHE["base_model"] = base_model
    MODEL_CACHE["tokenizer"] = tokenizer
    MODEL_CACHE["active_peft"] = peft_model
    MODEL_CACHE["active_subfolder"] = resolved_subfolder

    logger.info("✅ Hugging Face Model Loaded Successfully!")
    return peft_model, tokenizer

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    """Returns backend and model status."""
    import torch
    is_loaded = MODEL_CACHE["active_peft"] is not None
    return {
        "status": "online",
        "model_loaded": is_loaded,
        "active_subfolder": MODEL_CACHE["active_subfolder"],
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }

@app.post("/api/translate")
def translate(req: TranslationRequest):
    """Run neural machine translation on input text."""
    import torch

    model = MODEL_CACHE["active_peft"]
    tokenizer = MODEL_CACHE["tokenizer"]

    if model is None or tokenizer is None:
        # Fallback offline simulation if model loading was skipped or error
        logger.warning("Model not loaded yet. Lazy-loading default checkpoint...")
        try:
            model, tokenizer = initialize_hf_model(MODEL_CACHE["active_subfolder"])
        except Exception as e:
            logger.error(f"Failed to auto-load model: {e}")
            raise HTTPException(status_code=500, detail=f"Model not loaded: {str(e)}")

    prompt = f"<|im_start|>user\nTranslate {req.source_lang} to {req.target_lang}:\n{req.text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=max(req.temperature, 0.01),
            repetition_penalty=req.repetition_penalty,
            do_sample=(req.temperature > 0.05),
            pad_token_id=tokenizer.pad_token_id
        )

    latency_ms = round((time.time() - start_time) * 1000, 2)
    output_tokens = outputs[0][inputs.input_ids.shape[1]:]
    translation = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

    return {
        "source_text": req.text,
        "translation": translation,
        "source_lang": req.source_lang,
        "target_lang": req.target_lang,
        "latency_ms": latency_ms,
        "model_adapter": MODEL_CACHE["active_subfolder"]
    }

@app.post("/api/switch_model")
def switch_model(req: SwitchModelRequest):
    """Dynamically swap model adapter from Hugging Face subfolders."""
    from peft import PeftModel
    base_model = MODEL_CACHE["base_model"]

    if base_model is None:
        raise HTTPException(status_code=400, detail="Base model not initialized yet.")

    try:
        resolved_subfolder = resolve_subfolder(req.adapter_subfolder)
        token_val = os.environ.get("HF_TOKEN", None)
        logger.info(f"Switching adapter subfolder to '{resolved_subfolder}'...")
        new_peft = PeftModel.from_pretrained(
            base_model,
            HF_REPO_ID,
            subfolder=resolved_subfolder,
            token=token_val
        )
        new_peft.eval()

        MODEL_CACHE["active_peft"] = new_peft
        MODEL_CACHE["active_subfolder"] = resolved_subfolder

        return {
            "status": "success",
            "message": f"Successfully loaded adapter: {resolved_subfolder}",
            "active_subfolder": resolved_subfolder
        }
    except Exception as e:
        logger.error(f"Error switching adapter: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to switch adapter: {str(e)}")

# Mount static files (HTML, CSS, JS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Ekegusii NMT Server Running! Index HTML missing.</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
