from dotenv import load_dotenv
load_dotenv()

import os
import time
import torch
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from PIL import Image
from transformers import pipeline as hf_pipeline
from transformers import DonutProcessor, VisionEncoderDecoderModel
import subprocess
import threading
import uvicorn
import pathlib
import psutil

# ============================================
# Improve CUDA memory management (optional)
# ============================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================
# Load configuration from .env file
# ============================================
PUBLIC_IP = os.environ.get("PUBLIC_IP", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", "2727"))
FRONTEND_PORT = int(os.environ.get("FRONTEND_PORT", "2080"))
HF_HUB_TOKEN = os.environ.get("HF_HUB_TOKEN", "")
API_TOKEN = os.environ.get("API_TOKEN", "dev-token")

app = FastAPI(
    title="AI Image Generator & Analyzer API",
    description=(
        "API for image generation (Stable Diffusion, FLUX, Dreamlike), "
        "image analysis (BLIP Large, BLIP2 OPT 2.7b, Donut Base), "
        "and text-to-speech (TTS) with Kokoro."
    ),
    version="1.2.0"
)

# ============================================
# /config endpoint to supply API settings to the frontend
# ============================================
@app.get("/config")
async def get_config():
    return {
        "api_url": f"http://{PUBLIC_IP}:{API_PORT}",
        "api_token": f"Bearer {API_TOKEN}"
    }

# ============================================
# Enable CORS middleware
# ============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Choose device (GPU if available, else CPU)
# ============================================
device = "cuda" if torch.cuda.is_available() else "cpu"
device_index = 0 if torch.cuda.is_available() else -1

# ============================================
# Define project paths
# ============================================
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

# ============================================
# Utility function: free a port if occupied
# ============================================
def free_port(port: int):
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            for conn in proc.net_connections():
                if conn.laddr.port == port:
                    print(f"Killing process {proc.info['pid']} ({proc.info['name']}) on port {port}")
                    psutil.Process(proc.info["pid"]).terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue

# ============================================
# Token verification dependency
# ============================================
def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    token_value = authorization.split("Bearer ")[1]
    if token_value != API_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return token_value

# ============================================
# BLIP / BLIP2 analysis pipeline loader
# ============================================
def load_blip_pipeline(model_id: str):
    """
    Returns an image-to-text pipeline for BLIP or BLIP2 models.
    """
    return hf_pipeline("image-to-text", model=model_id, device=device_index)

# ============================================
# Donut analysis support (using VisionEncoderDecoderModel)
# ============================================
donut_processor = None
donut_model = None

def load_donut_model():
    """
    Loads the Donut Base processor and model once.
    """
    global donut_processor, donut_model
    if donut_processor is None or donut_model is None:
        donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base").to(device)
    return donut_processor, donut_model

def analyze_with_donut(img: Image.Image, task_prompt: str):
    """
    Processes an image with the Donut Base model using a task prompt.
    """
    processor, model = load_donut_model()

    # Optionally resize the image to limit memory usage
    max_size = 960
    w, h = img.size
    if w > max_size or h > max_size:
        ratio = min(max_size / w, max_size / h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)

    pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
    prompt_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=prompt_ids,
        max_new_tokens=128
    )
    seq = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return seq.strip()

# ============================================
# Analysis Models configuration
# ============================================
analysis_models = {
    "blip-large": {
        "type": "blip",
        "model_id": "Salesforce/blip-image-captioning-large"
    },
    "opt-2.7b": {
        "type": "blip",
        "model_id": "Salesforce/blip2-opt-2.7b"
    },
    "donut-base": {
        "type": "donut",
        "model_id": "naver-clova-ix/donut-base"
    },
}

# ============================================
# Endpoint: /analyze
# ============================================
@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    analysis_model: str = Query("blip-large"),
    prompt: str = Query(None, description="Task prompt (required for donut-base)"),
    token: str = Depends(verify_token)
):
    """
    Accepts an uploaded image and returns a caption or text output using the selected analysis model.
    For Donut Base, a task prompt is required.
    """
    if analysis_model not in analysis_models:
        raise HTTPException(status_code=400, detail=f"Invalid analysis model '{analysis_model}'")
    
    img = Image.open(file.file).convert("RGB")
    model_info = analysis_models[analysis_model]

    try:
        if model_info["type"] == "donut":
            if not prompt or prompt.strip() == "":
                raise HTTPException(status_code=400, detail="Prompt is required for Donut Base analysis")
            caption = analyze_with_donut(img, task_prompt=prompt)
        else:
            pipe = load_blip_pipeline(model_info["model_id"])
            result = pipe(img)
            del pipe
            torch.cuda.empty_cache()
            if isinstance(result, list) and result:
                caption = result[0].get("generated_text") or result[0].get("text") or str(result[0])
            else:
                caption = "No caption generated."
    except Exception as e:
        print(f"Analysis error for model={analysis_model}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        torch.cuda.empty_cache()
    return {"description": caption}

# ============================================
# Generation Models configuration
# ============================================
def load_generation_model(model: str):
    if model == "sd21":
        return StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
        ).to(device)
    elif model == "flux":
        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16
        )
        pipe.enable_model_cpu_offload()
        return pipe
    elif model == "dreamlike":
        return StableDiffusionPipeline.from_pretrained(
            "dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16
        ).to(device)
    else:
        raise HTTPException(status_code=400, detail="Invalid generation model")

# ============================================
# Endpoint: /generate
# ============================================
@app.post("/generate")
async def generate_image(
    prompt: str,
    model: str,
    token: str = Depends(verify_token)
):
    """
    Generates an image from a text prompt using the selected generation model.
    """
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

# ============================================
# Kokoro TTS loader and helper functions
# ============================================
def load_tts_kokoro(lang_code: str = "a"):
    """
    Loads the Kokoro TTS pipeline.
    Requires 'kokoro>=0.8.4' and system ffmpeg for MP3 conversion.
    """
    try:
        from kokoro import KPipeline
        return KPipeline(lang_code=lang_code)
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Kokoro not installed. Try 'pip install kokoro>=0.8.4 soundfile'."
        )

def flatten_kokoro_segment(segment):
    """
    Flattens a Kokoro audio segment to a 1D float32 NumPy array.
    """
    try:
        seg_array = np.array(segment, dtype=np.float32).flatten()
        return seg_array
    except Exception:
        flat_parts = []
        for item in segment:
            try:
                item_arr = np.array(item, dtype=np.float32).flatten()
                flat_parts.append(item_arr)
            except Exception:
                continue
        if not flat_parts:
            raise ValueError("Unable to flatten Kokoro segment; no valid items found.")
        return np.concatenate(flat_parts, axis=0)

def flatten_kokoro_output(generator):
    """
    Concatenates all audio segments from the Kokoro generator into one 1D float32 array.
    """
    all_segments = []
    for seg in generator:
        all_segments.append(flatten_kokoro_segment(seg))
    if not all_segments:
        raise ValueError("No audio segments from Kokoro pipeline.")
    return np.concatenate(all_segments, axis=0)

# ============================================
# Endpoint: /tts (Kokoro TTS only)
# ============================================
@app.get("/tts")
async def text_to_speech(
    text: str,
    token: str = Depends(verify_token),
    voice: str = "af_heart",
    fmt: str = Query("wav", description="Output audio format: 'wav' or 'mp3'")
):
    """
    Converts text to speech using the Kokoro TTS pipeline.
    Returns output in WAV or MP3 format.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    extension = "mp3" if fmt.lower() == "mp3" else "wav"
    filename = f"tts_{timestamp}.{extension}"
    output_path = TTS_OUTPUT_DIR / filename
    try:
        kokoro_pipe = load_tts_kokoro(lang_code="a")
        raw_output = kokoro_pipe(text, voice=voice)
        audio_data = flatten_kokoro_output(raw_output)
        sample_rate = 24000
        if extension == "wav":
            sf.write(str(output_path), audio_data, sample_rate)
        else:
            # Convert to MP3 using pydub and ffmpeg
            audio_segment = AudioSegment(
                data=audio_data.tobytes(),
                sample_width=4,  # 4 bytes for float32
                frame_rate=sample_rate,
                channels=1
            )
            audio_segment.export(str(output_path), format="mp3")
        msg = f"✅ Speech generated (Kokoro, {extension.upper()})"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "message": msg,
        "audio_url": f"http://{PUBLIC_IP}:{API_PORT}/files/tts/{filename}"
    }

# ============================================
# Free ports and run the server
# ============================================
if __name__ == "__main__":
    free_port(API_PORT)
    free_port(FRONTEND_PORT)
    def run_frontend_server():
        subprocess.Popen(["python3", "-m", "http.server", str(FRONTEND_PORT)], cwd=str(FRONTEND_DIR))
    frontend_thread = threading.Thread(target=run_frontend_server, daemon=True)
    frontend_thread.start()
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
