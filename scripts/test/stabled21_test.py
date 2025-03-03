import os
import time
import torch
from diffusers import StableDiffusionPipeline

def main():
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16
    ).to("cuda")
    
    prompt = "a futuristic cyberpunk city at sunset with neon lights"
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    
    output_dir = os.path.join(os.getcwd(), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"sd_output_{timestamp}.png")
    
    image.save(output_path)
    print(f"✅ Image generated and saved as {output_path}")

if __name__ == "__main__":
    main()
