import argparse
import os
import pyttsx3
import soundfile as sf
from transformers import pipeline

def tts_pyttsx3(text, output_path, voice="male", lang="en-US"):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    
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
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    print(f"Audio saved to: {output_path}")

def tts_transformers(text, output_path, model="facebook/fastspeech2-en-ljspeech", hf_token=None):
    print(f"Loading Transformers TTS pipeline: {model}")
    tts_pipe = pipeline("text-to-speech", model=model, use_auth_token=hf_token)
    result = tts_pipe(text)
    audio = result["audio"]
    sampling_rate = result["sampling_rate"]
    sf.write(output_path, audio, sampling_rate)
    print(f"Audio saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Test TTS: Convert text to speech and save as WAV.")
    parser.add_argument("--text", type=str, required=True, help="Text to convert to speech.")
    parser.add_argument("--backend", type=str, default="pyttsx3", choices=["pyttsx3", "transformers"],
                        help="TTS backend: 'pyttsx3' (system voice) or 'transformers' (Hugging Face model).")
    parser.add_argument("--voice", type=str, default="male", help="Desired voice (pyttsx3 only).")
    parser.add_argument("--lang", type=str, default="en-US", help="Desired language/accent (pyttsx3 only).")
    parser.add_argument("--model", type=str, default="facebook/fastspeech2-en-ljspeech",
                        help="Transformers TTS model (for transformers backend).")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (if required).")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV file path.")
    
    args = parser.parse_args()
    
    if args.backend == "pyttsx3":
        tts_pyttsx3(text=args.text, output_path=args.output, voice=args.voice, lang=args.lang)
    else:
        tts_transformers(text=args.text, output_path=args.output, model=args.model, hf_token=args.hf_token)

if __name__ == "__main__":
    main()
