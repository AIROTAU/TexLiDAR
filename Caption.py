"""
This script processes an image, splits it into four sections (left, front, right, back),
and generates detailed captions for each section using a pre-trained Florence-2 model.
The captions are then merged into a single output, which is displayed alongside the image.
"""

import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForCausalLM
from utils import split_image_into_sections, merge_captions


def run_example(task_prompt, image, model, processor, device):
    """Generate caption for a given image and task prompt."""
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


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Process an image and generate captions.")
    parser.add_argument("image_path", type=str, help="Path to the input image file")
    args = parser.parse_args()

    # --- Device Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Model & Processor Setup ---
    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype='auto'
    ).eval().to(device)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # --- Image Preprocessing ---
    # Open the image from the provided path
    image = Image.open(args.image_path)

    # Convert grayscale to RGB or 16-bit to 8-bit if needed
    if image.mode == 'L':  # Grayscale to RGB
        image = image.convert('RGB')
    elif image.mode == 'I;16':  # 16-bit to 8-bit
        image = image.point(lambda i: i * 255.0 / 65535.0).convert('RGB')

    # --- Image Splitting ---
    left, front, right, back = split_image_into_sections(image)

    # --- Caption Generation ---
    captions = [
        run_example('<DETAILED_CAPTION>', left, model, processor, device),
        run_example('<DETAILED_CAPTION>', front, model, processor, device),
        run_example('<DETAILED_CAPTION>', right, model, processor, device),
        run_example('<DETAILED_CAPTION>', back, model, processor, device)
    ]

    # --- Caption Merging ---
    merged_caption = merge_captions(captions)

    # --- Output ---
    print(f"Merged Caption:\n{merged_caption}")

    # Display the original image with the merged caption
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f'Merged Caption:\n{merged_caption}', fontsize=10, wrap=True)  # Display full caption
    plt.axis('off')  # Hide axes
    plt.show()


if __name__ == "__main__":
    main()
