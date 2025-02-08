"""
This script processes an image, splits it into four sections (left, front, right, back),
performs object detection on each section using a pre-trained Florence-2 model,
adjusts the detected ROIs to the full image coordinates, and displays the result
with bounding boxes drawn on the full image.
"""

import torch
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from utils import split_image_into_sections, adjust_roi_in_full_image, draw_bboxes
import argparse


def run_example(task_prompt, image, model, processor, device, text_input=None):
    """Run the model to generate captions or object detection results."""
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].to(device),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Process an image and generate captions.")
    parser.add_argument("image_path", type=str, help="Path to the input image file")
    args = parser.parse_args()

    # --- Device Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Model & Processor Setup ---
    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().to(device)
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
    sections = {'left': (split_image_into_sections(image))[0], 'front': (split_image_into_sections(image))[1],
                'right': (split_image_into_sections(image))[2], 'back': (split_image_into_sections(image))[3]}

    # Perform object detection on each section
    full_rois = []
    full_labels = []

    for direction, section in sections.items():
        task_prompt = '<DETAILED_CAPTION>'
        od_result = run_example(task_prompt, section, model=model, processor=processor, device=device)
        text_input = od_result[task_prompt]
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        od_result = run_example(task_prompt, section, text_input=text_input, model=model, processor=processor, device=device)
        for roi, label in zip(od_result['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'], od_result['<CAPTION_TO_PHRASE_GROUNDING>']['labels']):
            adjusted_rois = adjust_roi_in_full_image(roi, direction)

            for adj_roi in adjusted_rois:
                full_rois.append(adj_roi)
                full_labels.append(label)

    # Draw bounding boxes on the full image
    image = draw_bboxes(image, full_rois, full_labels)

    # Display the full image with all detected objects
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Object Detection on Full Image")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
