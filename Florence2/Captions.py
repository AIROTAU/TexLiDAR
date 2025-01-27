import os
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


# Function to run the model
def run_example(task_prompt, image):
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].to(device),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return generated_text


# Main logic
if __name__ == "__main__":
    input_folder = r"C:\temp\reflec\F"
    output_folder = r"C:\temp\reflec\F"
    os.makedirs(output_folder, exist_ok=True)

    captions_file = os.path.join(output_folder, "DETAILED_CAPTION.txt")
    with open(captions_file, "w") as f:
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_folder, filename)
                image = Image.open(image_path)

                if image.mode == 'L':
                    image = image.convert('RGB')
                if np.array(image).dtype == np.uint16:
                    image = image.point(lambda i: i * 255.0 / 65535.0).convert('RGB')

                # Generate captions
                for task_prompt in ['<DETAILED_CAPTION>']:
                    caption = run_example(task_prompt, image=image)
                    # Clean the generated caption by removing unwanted tokens
                    clean_caption = caption.replace("</s><s>", " ").replace("<s>", "").replace("</s>", "")

                    f.write(f"{filename} : [{task_prompt}] {clean_caption}\n")
