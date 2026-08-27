"""
Build Colab Cloudflare Free Deployment Notebook (.ipynb)
==========================================================
Generates a complete, standalone Jupyter Notebook for deploying the Ekegusii LLM Translation
model on Google Colab with Cloudflare Free Tunnel (cloudflared), Hugging Face Hub checkpoints,
and GitHub HTML/CSS frontend assets.
"""

import json
import os

def create_notebook():
    cells = []

    # ---------------------------------------------------------------------------
    # Cell 1: Markdown Title & Documentation
    # ---------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🌍 Ekegusii Multilingual NMT: Google Colab Cloudflare Free Tunnel Deployment\n",
            "This notebook provides a 1-click solution to deploy the fine-tuned **Ekegusii LLM Translation Model** on **Google Colab** using **Cloudflare Free Tunnel (`cloudflared`)**, **Hugging Face Hub** (`aykgeh/Ekegusii-LLM-Translation`), and **GitHub** for the HTML/CSS/JS web interface.\n",
            "\n",
            "### 🚀 Key Features:\n",
            "- **🤗 Model Source**: Loaded directly from Hugging Face Repo `aykgeh/Ekegusii-LLM-Translation` (Supports checkpoint subfolder `qwen/E1_English_Ekegusii/checkpoint-8000` & E10 Winner).\n",
            "- **⚡ Cloudflare Free Tunnel**: Exposes your Colab GPU web server via a public, secure `https://*.trycloudflare.com` URL (No sign-up, no password, 100% free).\n",
            "- **🎨 GitHub Web Assets**: Uses clean HTML5/CSS3 glassmorphism web interface (`index.html`, `style.css`, `app.js`) fetched from GitHub.\n",
            "- **⚡ 4-Bit NF4 Quantization**: Runs on standard free Google Colab T4 GPU with ~4.5 GB VRAM usage."
        ]
    })

    # ---------------------------------------------------------------------------
    # Cell 2: Step 1 - Check GPU Runtime
    # ---------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🖥️ Step 1: Verify Google Colab GPU Environment\n",
            "Ensure you have enabled GPU accelerator in Colab (`Runtime` -> `Change runtime type` -> `T4 GPU`)."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import sys\n",
            "\n",
            "print('=' * 60)\n",
            "print('🔍 CHECKING COLAB GPU ACCELERATOR...')\n",
            "print('=' * 60)\n",
            "print(f'PyTorch Version : {torch.__version__}')\n",
            "print(f'CUDA Available  : {torch.cuda.is_available()}')\n",
            "\n",
            "if torch.cuda.is_available():\n",
            "    device_name = torch.cuda.get_device_name(0)\n",
            "    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)\n",
            "    print(f'✅ GPU Detected   : {device_name} ({vram_gb} GB VRAM)')\n",
            "else:\n",
            "    print('⚠️ WARNING: No GPU detected! Go to Runtime -> Change runtime type -> Select T4 GPU.')"
        ]
    })

    # ---------------------------------------------------------------------------
    # Cell 3: Step 2 - Hugging Face Authentication
    # ---------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔑 Step 2: Hugging Face Authentication\n",
            "This step reads `HF_TOKEN` from Google Colab Secrets (the 🔑 icon on the left sidebar) or allows manual token entry."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "from huggingface_hub import login\n",
            "\n",
            "#@title 🔑 Hugging Face Token Setup { run: 'auto' }\n",
            "HF_TOKEN_INPUT = \"\" #@param {type:\"string\"}\n",
            "\n",
            "hf_token = None\n",
            "# 1. Try reading from Colab Secrets\n",
            "try:\n",
            "    from google.colab import userdata\n",
            "    try:\n",
            "        hf_token = userdata.get('HF_TOKEN')\n",
            "    except Exception:\n",
            "        hf_token = None\n",
            "except Exception:\n",
            "    hf_token = None\n",
            "\n",
            "# 2. Fallback to manual entry\n",
            "if not hf_token and HF_TOKEN_INPUT.strip():\n",
            "    hf_token = HF_TOKEN_INPUT.strip()\n",
            "\n",
            "if hf_token:\n",
            "    login(token=hf_token.strip())\n",
            "    os.environ[\"HF_TOKEN\"] = hf_token.strip()\n",
            "    print(\"✅ Hugging Face Authentication Successful!\")\n",
            "else:\n",
            "    print(\"ℹ️ Running in Public Model mode (No HF_TOKEN required for public repos).\")"
        ]
    })

    # ---------------------------------------------------------------------------
    # Cell 4: Step 3 - Install Packages & Download Cloudflare Tunnel Binary
    # ---------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📦 Step 3: Install Required Packages & Download Cloudflare Tunnel (`cloudflared`)\n",
            "We install `transformers`, `peft`, `bitsandbytes`, `fastapi`, `uvicorn`, and download the official **Cloudflare `cloudflared`** Linux binary."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Install Python ML & Web dependencies\n",
            "!pip install -q transformers peft bitsandbytes accelerate fastapi uvicorn pydantic streamlit nest_asyncio requests\n",
            "\n",
            "# 2. Download and install Cloudflare Tunnel (cloudflared) debian package\n",
            "print('⚡ Downloading Cloudflare Tunnel (cloudflared)...')\n",
            "!wget -q -O cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb\n",
            "!dpkg -i cloudflared.deb\n",
            "!cloudflared --version\n",
            "print('✅ Cloudflare Tunnel Binary Installed Successfully!')"
        ]
    })

    # ---------------------------------------------------------------------------
    # Cell 5: Step 4 - Clone GitHub Repository & Web Assets
    # ---------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📁 Step 4: Clone GitHub Repository (HTML, CSS, JS Frontend Assets)\n",
            "We fetch the project repository from GitHub to access `web/static/index.html`, `web/static/style.css`, `web/static/app.js`, and `web/server.py`."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "\n",
            "# Clone repository if running in Colab\n",
            "REPO_URL = \"https://github.com/aykahsay/Ekegusii-LLM-Translation.git\"\n",
            "\n",
            "if not os.path.exists('web'):\n",
            "    print('📥 Cloning project repository from GitHub...')\n",
            "    !git clone {REPO_URL} repo_temp\n",
            "    if os.path.exists('repo_temp/web'):\n",
            "        !cp -r repo_temp/web ./web\n",
            "        !cp -r repo_temp/app.py ./app.py 2>/dev/null || true\n",
            "        !rm -rf repo_temp\n",
            "        print('✅ Web assets successfully extracted from GitHub!')\n",
            "    else:\n",
            "        print('⚠️ Using local web directory.')\n",
            "else:\n",
            "    print('✅ Directory `web/` already exists.')"
        ]
    })

    # ---------------------------------------------------------------------------
    # Cell 6: Step 5 - Load Hugging Face Model Checkpoint
    # ---------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🤗 Step 5: Load Base Model & Hugging Face Checkpoint Adapter\n",
            "We load **`Qwen/Qwen2.5-7B-Instruct`** in 4-bit quantization and attach the fine-tuned LoRA checkpoint adapter from **`aykgeh/Ekegusii-LLM-Translation`**.\n",
            "\n",
            "> 📍 **Specified Checkpoint**: `qwen/E1_English_Ekegusii/checkpoint-8000`\n",
            "> 🏆 **Winner Model Option**: `qwen/E10_Model_B_English_Ekegusii`"
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch, os\n",
            "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
            "from peft import PeftModel\n",
            "\n",
            "# Model identifiers\n",
            "BASE_MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'\n",
            "HF_REPO_ID = 'aykgeh/Ekegusii-LLM-Translation'\n",
            "# Default checkpoint requested by user (E10 Winner Model)\n",
            "CHECKPOINT_SUBFOLDER = 'qwen/E10_Model_B_English_Ekegusii/checkpoint-8000'\n",
            "\n",
            "token_value = os.environ.get('HF_TOKEN', None)\n",
            "device_map = {\"\": 0} if torch.cuda.is_available() else \"auto\"\n",
            "\n",
            "print('⏳ 1/3 Loading 4-bit Quantization Config...')\n",
            "bnb_config = BitsAndBytesConfig(\n",
            "    load_in_4bit=True,\n",
            "    bnb_4bit_quant_type='nf4',\n",
            "    bnb_4bit_use_double_quant=True,\n",
            "    bnb_4bit_compute_dtype=torch.bfloat16,\n",
            "    llm_int8_enable_fp32_cpu_offload=True\n",
            ")\n",
            "\n",
            "print('⏳ 2/3 Loading Qwen2.5-7B Base Model & Tokenizer...')\n",
            "tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, padding_side='left', token=token_value)\n",
            "if tokenizer.pad_token is None:\n",
            "    tokenizer.pad_token = tokenizer.eos_token\n",
            "\n",
            "base_model = AutoModelForCausalLM.from_pretrained(\n",
            "    BASE_MODEL_ID,\n",
            "    quantization_config=bnb_config,\n",
            "    device_map=device_map,\n",
            "    torch_dtype=torch.bfloat16,\n",
            "    token=token_value\n",
            ")\n",
            "\n",
            "print(f\"⏳ 3/3 Attaching Adapter Checkpoint ('{CHECKPOINT_SUBFOLDER}') from Hugging Face Hub...\")\n",
            "peft_model = PeftModel.from_pretrained(\n",
            "    base_model,\n",
            "    HF_REPO_ID,\n",
            "    subfolder=CHECKPOINT_SUBFOLDER,\n",
            "    token=token_value\n",
            ")\n",
            "peft_model.eval()\n",
            "\n",
            "# Register model in server cache (Bulletproof web module resolution)\n",
            "import sys, os\n",
            "\n",
            "if not any(os.path.exists(os.path.join(p, 'web')) for p in ['.', 'repo', '/content/repo', os.getcwd()]):\n",
            "    print('📥 Cloning project repository from GitHub...')\n",
            "    !git clone https://github.com/aykahsay/Ekegusii-LLM-Translation.git repo\n",
            "\n",
            "for candidate in [os.getcwd(), os.path.abspath('repo'), '/content/repo', '/content']:\n",
            "    if os.path.exists(os.path.join(candidate, 'web')):\n",
            "        if candidate not in sys.path:\n",
            "            sys.path.insert(0, candidate)\n",
            "\n",
            "from web.server import MODEL_CACHE\n",
            "MODEL_CACHE['base_model'] = base_model\n",
            "MODEL_CACHE['tokenizer'] = tokenizer\n",
            "MODEL_CACHE['active_peft'] = peft_model\n",
            "MODEL_CACHE['active_subfolder'] = CHECKPOINT_SUBFOLDER\n",
            "\n",
            "print('✅ Hugging Face Checkpoint Successfully Loaded on GPU and Registered in Web Cache!')"
        ]
    })

    # ---------------------------------------------------------------------------
    # Cell 7: Step 6 - Launch FastAPI Server in Background
    # ---------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🚀 Step 6: Launch FastAPI Web Backend\n",
            "Starts the Uvicorn web server in a background process listening on `http://127.0.0.1:8000`."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import subprocess, time\n",
            "\n",
            "print('🌐 Starting FastAPI Web Server on port 8000...')\n",
            "server_process = subprocess.Popen(\n",
            "    ['python', '-m', 'uvicorn', 'web.server:app', '--host', '127.0.0.1', '--port', '8000'],\n",
            "    stdout=subprocess.PIPE,\n",
            "    stderr=subprocess.PIPE\n",
            ")\n",
            "\n",
            "time.sleep(3)\n",
            "print('✅ FastAPI Backend Server active on port 8000!')"
        ]
    })

    # ---------------------------------------------------------------------------
    # Cell 8: Step 7 - Launch Cloudflare Tunnel & Extract URL
    # ---------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## ⚡ Step 7: Launch Cloudflare Free Tunnel (`cloudflared`) & Get Public HTTPS Link\n",
            "Cloudflare Tunnel establishes an encrypted bridge to your Colab server and provides a free temporary public URL (`https://*.trycloudflare.com`)."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import subprocess, re, time\n",
            "from IPython.display import HTML, display\n",
            "\n",
            "# Start cloudflared tunnel in background\n",
            "tunnel_process = subprocess.Popen(\n",
            "    ['cloudflared', 'tunnel', '--url', 'http://127.0.0.1:8000'],\n",
            "    stdout=subprocess.PIPE,\n",
            "    stderr=subprocess.STDOUT,\n",
            "    text=True\n",
            ")\n",
            "\n",
            "print('⚡ Initializing Cloudflare Tunnel...')\n",
            "public_url = None\n",
            "\n",
            "# Wait and parse cloudflared output for the trycloudflare URL\n",
            "for i in range(30):\n",
            "    line = tunnel_process.stdout.readline()\n",
            "    if not line:\n",
            "        time.sleep(0.5)\n",
            "        continue\n",
            "    match = re.search(r'https://[a-zA-Z0-9-]+\\.trycloudflare\\.com', line)\n",
            "    if match:\n",
            "        public_url = match.group(0)\n",
            "        break\n",
            "\n",
            "if public_url:\n",
            "    display(HTML(f\"\"\"\n",
            "    <div style=\"background: linear-gradient(135deg, #0f172a, #1e293b); padding: 24px; border-radius: 16px; border: 2px solid #3b82f6; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.5); font-family: sans-serif;\">\n",
            "        <h2 style=\"color: #60a5fa; margin-bottom: 8px;\">⚡ Cloudflare Live Tunnel Active!</h2>\n",
            "        <p style=\"color: #cbd5e1; font-size: 1.1em;\">Click the link below to open your deployed Ekegusii NMT Web Application:</p>\n",
            "        <a href=\"{public_url}\" target=\"_blank\" style=\"display: inline-block; background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; padding: 14px 28px; border-radius: 10px; font-weight: bold; font-size: 1.2em; text-decoration: none; margin: 16px 0; box-shadow: 0 4px 15px rgba(59,130,246,0.4);\">\n",
            "            🌐 Open Web Portal: {public_url}\n",
            "        </a>\n",
            "        <p style=\"color: #94a3b8; font-size: 0.85em; margin-top: 10px;\">\n",
            "            🤗 Hugging Face Model Repo: <strong>aykgeh/Ekegusii-LLM-Translation</strong><br>\n",
            "            📍 Loaded Checkpoint: <strong>{CHECKPOINT_SUBFOLDER}</strong>\n",
            "        </p>\n",
            "    </div>\n",
            "    \"\"\"))\n",
            "else:\n",
            "    print('⚠️ Tunnel setup delayed. Checking tunnel process log...')\n",
            "    print(tunnel_process.stdout.read())"
        ]
    })

    # ---------------------------------------------------------------------------
    # Cell 9: Step 8 - In-Notebook Quick Test
    # ---------------------------------------------------------------------------
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🧪 Step 8: Live Interactive Translation Sandbox (Inside Notebook)\n",
            "You can test sentence translation directly inside Colab notebook cells while your Cloudflare web application is running."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def translate_sentence(text: str, source_lang: str = 'English', target_lang: str = 'Ekegusii') -> str:\n",
            "    \"\"\"Translate text directly using loaded PyTorch model.\"\"\"\n",
            "    prompt = f'<|im_start|>user\\nTranslate {source_lang} to {target_lang}:\\n{text}<|im_end|>\\n<|im_start|>assistant\\n'\n",
            "    inputs = tokenizer(prompt, return_tensors='pt').to(peft_model.device)\n",
            "    \n",
            "    with torch.no_grad():\n",
            "        outputs = peft_model.generate(\n",
            "            **inputs,\n",
            "            max_new_tokens=128,\n",
            "            temperature=0.1,\n",
            "            do_sample=False,\n",
            "            pad_token_id=tokenizer.pad_token_id\n",
            "        )\n",
            "    \n",
            "    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()\n",
            "\n",
            "# --- Test Public Service Announcement (PSA) ---\n",
            "test_sentence = \"Please wash your hands regularly with clean running water and soap to prevent cholera infection.\"\n",
            "result = translate_sentence(test_sentence, source_lang=\"English\", target_lang=\"Ekegusii\")\n",
            "\n",
            "print('=' * 70)\n",
            "print(f'INPUT (EN):  {test_sentence}')\n",
            "print(f'OUTPUT (EKE): {result}')\n",
            "print('=' * 70)"
        ]
    })

    notebook_content = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "colab_cloudflare_deployment.ipynb",
                "provenance": []
            },
            "gpuClass": "standard",
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # Write notebook files
    paths_to_write = [
        "c:/Users/Admin/OneDrive - United States International University (USIU)/Documents/NLP/Multilogual_transaltion_nlp/notebooks/colab_cloudflare_deployment.ipynb",
        "c:/Users/Admin/OneDrive - United States International University (USIU)/Documents/NLP/Multilogual_transaltion_nlp/colab_cloudflare_deployment.ipynb"
    ]

    for p in paths_to_write:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(notebook_content, f, indent=2)
        print(f"Created Notebook: {p}")

if __name__ == '__main__':
    create_notebook()
