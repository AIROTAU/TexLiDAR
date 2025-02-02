import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForCausalLM
from utils import split_image_into_sections, merge_captions

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


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

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process an image and generate captions.")
parser.add_argument(
    "image_path",
    type=str,
    nargs="?",  # Makes the argument optional
    default="C:/TexLiDAR/Images/8344.png",  # Sets default to specific image path
    help="Path to the input image file "
)
args = parser.parse_args()

# Open the image from the provided path
image = Image.open(args.image_path)

# Convert grayscale to RGB
if image.mode == 'L':
    image = image.convert('RGB')

# Convert 16-bit image to 8-bit if needed
image_np = np.array(image)
if image_np.dtype == np.uint16:
    image = image.point(lambda i: i * 255.0 / 65535.0).convert('RGB')

# Split the image into four sections
left, front, right, back = split_image_into_sections(image)

# Generate captions for each direction
captions = [run_example('<DETAILED_CAPTION>', left), run_example('<DETAILED_CAPTION>', front),
            run_example('<DETAILED_CAPTION>', right), run_example('<DETAILED_CAPTION>', back)]

# Merge the captions into one
merged_caption = merge_captions(captions)

# Print the final merged caption
print(f"Merged Caption:\n{merged_caption}")

# Display the original image with the merged caption
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title(f'Merged Caption:\n{merged_caption}', fontsize=10, wrap=True)  # Display full caption
plt.axis('off')  # Hide axes
plt.show()
