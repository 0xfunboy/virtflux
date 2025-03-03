import os
import torch
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    TrOCRProcessor, VisionEncoderDecoderModel
)

def main():
    image_path = os.path.join(os.getcwd(), "..", "inputs", "test_image.jpg")
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    image = Image.open(image_path).convert("RGB")

    print("=== Generating caption with BLIP ===")
    blip_model_id = "Salesforce/blip-image-captioning-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    blip_processor = BlipProcessor.from_pretrained(blip_model_id)
    blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_id).to(device)
    
    blip_inputs = blip_processor(images=image, return_tensors="pt").to(device)
    blip_out = blip_model.generate(**blip_inputs)
    caption = blip_processor.decode(blip_out[0], skip_special_tokens=True)
    print(f"BLIP Caption: {caption}")

    print("=== OCR with TrOCR ===")
    trocr_model_id = "microsoft/trocr-base-stage1"
    
    trocr_processor = TrOCRProcessor.from_pretrained(trocr_model_id)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_model_id).to(device)
    
    pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = trocr_model.generate(pixel_values)
    ocr_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"TrOCR Text: {ocr_text}")

if __name__ == "__main__":
    main()
