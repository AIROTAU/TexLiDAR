import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer


# Function to plot bounding boxes on the image
def plot_bbox(image, data):
    img_width, img_height = image.size
    fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100), dpi=100)
    ax.imshow(image)

    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    ax.axis('off')
    plt.show()


if __name__ == "__main__":
    # Object Detection
    # Load a single image
    image_path = r"C:\temp\FDenoisedSplitted\1_image_003.png"  # Update this path
    image = Image.open(image_path)

    # Convert grayscale to RGB if needed
    if image.mode == 'L':
        image = image.convert('RGB')

    # Normalize uint16 images to 8-bit if necessary
    if np.array(image).dtype == np.uint16:
        image = image.point(lambda i: i * 255.0 / 65535.0).convert('RGB')
    od_results = run_example('<OD>', image=image)
    print("Object Detection Results:", od_results)
    plot_bbox(image, od_results['<OD>'])

    # Captions
    for task_prompt in ['<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>']:
        caption = run_example(task_prompt, image=image)
        print(f"{task_prompt}:\n{'-' * len(task_prompt)}")
        print(f"{caption[f'{task_prompt}']}\n")
