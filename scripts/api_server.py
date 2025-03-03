from dotenv import load_dotenv
load_dotenv()

import os
import time
import torch
import pyttsx3
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from PIL import Image
from transformers import pipeline as hf_pipeline
import subprocess
import threading
import uvicorn
import pathlib
import psutil

# Improve CUDA memory management (optional)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load configuration from .env
# PUBLIC_IP is used only for generating external URLs in responses
PUBLIC_IP = os.environ.get("PUBLIC_IP", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", "2727"))
FRONTEND_PORT = int(os.environ.get("FRONTEND_PORT", "2080"))
HF_HUB_TOKEN = os.environ.get("HF_HUB_TOKEN", "")
API_TOKEN = os.environ.get("API_TOKEN", "dev-token")

app = FastAPI(
    title="AI Image Generator & Analyzer API",
    description="API for image generation, image analysis (BLIP/OmniParser), and text-to-speech (TTS).",
    version="1.2.0"
)

# Expose a /config endpoint to supply API configuration to the frontend.
@app.get("/config")
async def get_config():
    return {
        "api_url": f"http://{PUBLIC_IP}:{API_PORT}",
        "api_token": f"Bearer {API_TOKEN}"
    }

# Enable CORS (for production, restrict origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use GPU if available; fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
device_index = 0 if torch.cuda.is_available() else -1

###############################################################################
# Determine project paths relative to this file
###############################################################################
# Since this file is located in /home/funboy/virtflux/scripts/api_server.py,
# the project root is one level above the "scripts" folder.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
TTS_OUTPUT_DIR = OUTPUTS_DIR / "tts"
TTS_OUTPUT_DIR.mkdir(exist_ok=True)
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if not FRONTEND_DIR.exists():
    raise FileNotFoundError(f"Frontend directory not found at {FRONTEND_DIR}")

# Mount /files to serve generated images and audio
app.mount("/files", StaticFiles(directory=str(OUTPUTS_DIR)), name="files")

###############################################################################
# Utility: Free a port if occupied
###############################################################################
def free_port(port):
    """Forcefully free a port if it is occupied."""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections():
                if conn.laddr.port == port:
                    print(f"Killing process {proc.info['pid']} ({proc.info['name']}) using port {port}")
                    psutil.Process(proc.info['pid']).terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue

###############################################################################
# Token Verification
###############################################################################
def verify_token(authorization: str = Header(None)):
    """Verify the Bearer token in the request header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    token_value = authorization.split("Bearer ")[1]
    if token_value != API_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return token_value

###############################################################################
# Analysis Models Configuration
###############################################################################
analysis_models = {
    "blip-large": {
        "task": "image-to-text",
        "model_id": "Salesforce/blip-image-captioning-large"
    },
    "opt-2.7b": {
        "task": "image-to-text",
        "model_id": "Salesforce/blip2-opt-2.7b"
    },
    "omnparser": {
        "task": "image-text-to-text",
        "model_id": "microsoft/OmniParser"
    }
}

def load_analysis_pipeline(model_key: str):
    """Lazy load the image analysis pipeline.
       Note: 'use_auth_token' is omitted because it causes a TypeError for image-to-text pipelines.
    """
    if model_key not in analysis_models:
        raise HTTPException(status_code=400, detail=f"Invalid analysis model '{model_key}'")
    config = analysis_models[model_key]
    return hf_pipeline(
        config["task"],
        model=config["model_id"],
        device=device_index
    )

###############################################################################
# Lazy Loading for Image Generation Models
###############################################################################
def load_generation_model(model: str):
    """Lazy load an image generation model."""
    if model == "sd21":
        return StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16
        ).to(device)
    elif model == "flux":
        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()
        return pipe
    elif model == "dreamlike":
        return StableDiffusionPipeline.from_pretrained(
            "dreamlike-art/dreamlike-photoreal-2.0",
            torch_dtype=torch.float16
        ).to(device)
    else:
        raise HTTPException(status_code=400, detail="Invalid generation model")

###############################################################################
# Lazy Loading for Transformers TTS (CPU)
###############################################################################
def load_tts_transformers(model: str = "facebook/fastspeech2-en-ljspeech"):
    """Lazy load the Transformers TTS pipeline."""
    try:
        return hf_pipeline(
            "text-to-speech",
            model=model,
            device=-1,
            use_auth_token=HF_HUB_TOKEN
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Transformers loading error: {str(e)}")

###############################################################################
# API Endpoint: /analyze
###############################################################################
@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    analysis_model: str = Query("blip-large"),
    token: str = Depends(verify_token)
):
    """Analyze an image and return a generated caption."""
    pipe = load_analysis_pipeline(analysis_model)
    try:
        img = Image.open(file.file).convert("RGB")
        result = pipe(img)
        if isinstance(result, list) and result:
            if "generated_text" in result[0]:
                caption = result[0]["generated_text"]
            elif "text" in result[0]:
                caption = result[0]["text"]
            else:
                caption = str(result[0])
        else:
            caption = "No caption generated."
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        del pipe
        torch.cuda.empty_cache()
    return {"description": caption}

###############################################################################
# API Endpoint: /generate
###############################################################################
@app.post("/generate")
async def generate_image(
    prompt: str,
    model: str,
    token: str = Depends(verify_token)
):
    """Generate an image based on a text prompt using the specified model."""
    pipe = load_generation_model(model)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"generated_{model}_{timestamp}.png"
    output_path = OUTPUTS_DIR / filename
    try:
        if model == "flux":
            image = pipe(prompt, num_inference_steps=4).images[0]
        else:
            image = pipe(prompt).images[0]
        image.save(str(output_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        del pipe
        torch.cuda.empty_cache()
    return {
        "message": "✅ Image generated",
        "image_url": f"http://{PUBLIC_IP}:{API_PORT}/files/{filename}"
    }

###############################################################################
# API Endpoint: /tts
###############################################################################
@app.get("/tts")
async def text_to_speech(
    text: str,
    token: str = Depends(verify_token),
    tts_backend: str = "pyttsx3",
    voice: str = "male",
    lang: str = "en-US"
):
    """Convert text to speech using the specified TTS backend."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"tts_{timestamp}.wav"
    output_path = TTS_OUTPUT_DIR / filename
    if tts_backend == "transformers":
        tts_pipe = load_tts_transformers()
        try:
            result = tts_pipe(text)
            audio = result["audio"]
            sr = result["sampling_rate"]
            sf.write(str(output_path), audio, sr)
            msg = "✅ Speech generated (transformers)"
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            del tts_pipe
            torch.cuda.empty_cache()
        return {
            "message": msg,
            "audio_url": f"http://{PUBLIC_IP}:{API_PORT}/files/tts/{filename}"
        }
    elif tts_backend == "pyttsx3":
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            selected_voice_id = None
            for v in voices:
                if voice.lower() in v.name.lower() or voice.lower() in v.id.lower():
                    if lang.lower() in str(v.languages).lower() or lang.lower() in v.name.lower():
                        selected_voice_id = v.id
                        break
            if not selected_voice_id:
                for v in voices:
                    if voice.lower() in v.name.lower() or voice.lower() in v.id.lower():
                        selected_voice_id = v.id
                        break
            if not selected_voice_id and voices:
                selected_voice_id = voices[0].id
            engine.setProperty('voice', selected_voice_id)
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {
            "message": "✅ Speech generated (pyttsx3)",
            "audio_url": f"http://{PUBLIC_IP}:{API_PORT}/files/tts/{filename}"
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid TTS backend specified.")

###############################################################################
# Free ports before starting the servers
###############################################################################
def free_port(port):
    """Forcefully free a port if it is occupied."""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections():
                if conn.laddr.port == port:
                    print(f"Killing process {proc.info['pid']} ({proc.info['name']}) using port {port}")
                    psutil.Process(proc.info['pid']).terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue

###############################################################################
# Start both API and the static frontend server
###############################################################################
if __name__ == "__main__":
    # Free required ports before starting
    free_port(API_PORT)
    free_port(FRONTEND_PORT)

    # Start the static server to serve the frontend
    def run_frontend_server():
        subprocess.Popen(["python3", "-m", "http.server", str(FRONTEND_PORT)], cwd=str(FRONTEND_DIR))
    
    frontend_thread = threading.Thread(target=run_frontend_server, daemon=True)
    frontend_thread.start()

    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
