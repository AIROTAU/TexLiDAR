import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from utils import split_image_into_sections, adjust_roi_in_full_image, draw_bboxes
import argparse

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


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

# Split image into four sections
sections = {'left': (split_image_into_sections(image))[0], 'front': (split_image_into_sections(image))[1],
            'right': (split_image_into_sections(image))[2], 'back': (split_image_into_sections(image))[3]}

# Perform object detection on each section
full_rois = []
full_labels = []

for direction, section in sections.items():

    task_prompt = '<DETAILED_CAPTION>'
    od_result = run_example(task_prompt, section)
    print(od_result)
    text_input = od_result[task_prompt]
    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    od_result = run_example(task_prompt, section, text_input)
    print(od_result)
    for roi, label in zip(od_result['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes'], od_result['<CAPTION_TO_PHRASE_GROUNDING>']['labels']):
        adjusted_rois = adjust_roi_in_full_image(roi, direction)  # This may return 1 or 2 ROIs

        for adj_roi in adjusted_rois:  # Handle both normal and split cases
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
