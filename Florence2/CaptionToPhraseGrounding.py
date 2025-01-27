import os
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw, ImageFont

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def filter_bboxes(data, min_size=20, max_size=500):
    """
    Filters bounding boxes based on size thresholds.

    Parameters:
    - data (dict): Dictionary containing 'bboxes' and 'labels'.
    - min_size (int): Minimum width/height of bounding boxes to retain.
    - max_size (int): Maximum width/height of bounding boxes to retain.

    Returns:
    - dict: Filtered data containing only bounding boxes within the size range.
    """
    filtered_bboxes = []
    filtered_labels = []

    for (x1, y1, x2, y2), label in zip(data['bboxes'], data['labels']):
        width = x2 - x1
        height = y2 - y1

        # Check if the bounding box size is within the thresholds
        if min_size <= width <= max_size and min_size <= height <= max_size:
            filtered_bboxes.append((x1, y1, x2, y2))
            filtered_labels.append(label)

    return {'bboxes': filtered_bboxes, 'labels': filtered_labels}


# Function to run the model
def run_example(task_prompt, image, text_input=None):
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


def save_image_with_bbox(image, data, output_path):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for (x1, y1, x2, y2), label in zip(data['bboxes'], data['labels']):
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1, x1 + text_width + 4, y1 + text_height + 4], fill="red")
        draw.text((x1 + 2, y1 + 2), label, fill="white", font=font)

    image.save(output_path)


# Main logic
if __name__ == "__main__":
    input_folder = r"C:\temp\ambient\FSplittedDenoised"  # Update to your input folder
    output_folder = r"C:\temp\ambient\FSplittedDenoised\OD\person"  # Update to your object detection output folder
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            if image.mode == 'L':
                image = image.convert('RGB')
            if np.array(image).dtype == np.uint16:
                image = image.point(lambda i: i * 255.0 / 65535.0).convert('RGB')

            # Object Detection
            od_results = run_example('<CAPTION_TO_PHRASE_GROUNDING>', image=image, text_input="A person.")
            filtered_results = filter_bboxes(od_results['<CAPTION_TO_PHRASE_GROUNDING>'], min_size=20, max_size=500)

            # Save filtered results
            output_image_path = os.path.join(output_folder, f"{filename}")
            save_image_with_bbox(image, filtered_results, output_image_path)
