"""
test_omniparser.py

This script tests the Microsoft OmniParser v2.0 model for image-text-to-text tasks.
It loads a sample image, passes it along with a prompt to the pipeline, and prints the output.
"""

import os
from PIL import Image
from transformers import pipeline

def main():
    # Define the local model path.
    model_path = "/home/funboy/virtflux/models/models--microsoft--OmniParser-v2.0"
    
    # Set device: use GPU (0) if available, otherwise CPU (-1)
    device = 0 if os.environ.get("CUDA_VISIBLE_DEVICES", None) is not None else -1
    
    # Create the pipeline for the "image-text-to-text" task using the local model.
    try:
        parser_pipeline = pipeline("image-text-to-text", model=model_path, device=device)
    except Exception as e:
        print("Error loading OmniParser-v2.0 pipeline:", e)
        return

    # Define a test image path (adjust as necessary)
    image_path = "/home/funboy/virtflux/inputs/test.jpg"
    if not os.path.exists(image_path):
        print(f"Test image not found at {image_path}. Please place a valid test image there.")
        return
    
    # Open and convert the image to RGB.
    image = Image.open(image_path).convert("RGB")
    
    # Define a prompt to guide the model; for example, "Describe the image."
    prompt = "Describe the image."
    
    # Run the pipeline with the image and prompt.
    try:
        result = parser_pipeline(image, prompt)
    except Exception as e:
        print("Error during pipeline inference:", e)
        return
    
    # Print the result.
    print("Output from OmniParser-v2.0:")
    print(result)

if __name__ == "__main__":
    main()
