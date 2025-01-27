import random
import re

# Prior knowledge about directions
directions = ["left", "front", "right", "back"]

# List of prompt variations
prompt_pool = [
    "The view from the {direction} shows",
    "From the {direction} angle, you can see",
    "Looking towards the {direction}, we see",
    "Seen from the {direction}, the image features",
    "When viewed from the {direction}, the scene depicts",
    "Facing the {direction}, the image highlights",
    "Captured from the {direction}, this image shows",
    "The {direction} perspective reveals",
    "From the {direction} perspective, we observe",
    "As seen from the {direction}, the image presents",
    "From the {direction} viewpoint, we can observe",
    "Looking in the {direction}, the image portrays",
    "Captured from the {direction} angle, the scene depicts",
    "Seen from the {direction} perspective, we see",
    "The {direction} view reveals",
    "From the {direction} side, the image shows",
    "Through the {direction} lens, we observe",
    "The scene viewed from the {direction} highlights",
    "With the {direction}-facing angle, we can see",
    "In the {direction} direction, the image presents",
    "The {direction}-facing perspective showcases",
    "Captured from the {direction}-ward view, we see",
    "When viewed from the {direction}, this image reveals",
    "Peering towards the {direction}, the image shows",
    "In the {direction} direction, the photo captures",
    "The {direction} shot reveals",
    "As viewed from the {direction}, the scene portrays",
    "The {direction} view offers a glimpse of",
    "The {direction}-oriented perspective showcases",
    "From the {direction}-facing view, we can observe"
]


# Function to clean the caption by removing unwanted prefixes
def clean_caption_prefix(captions):
    cleaned_captions = []

    # Define the patterns to remove
    patterns_to_remove = [
        r"^\[<DETAILED_CAPTION>\]\s*The image shows",
        r"a black and white photo of"
    ]

    for caption in captions:
        for pattern in patterns_to_remove:
            caption = re.sub(pattern, "", caption)  # Remove the pattern using regex substitution
        cleaned_captions.append(caption.strip())  # Strip any leading/trailing spaces

    return cleaned_captions


# Function to clean and add directions with dynamic prompts
def clean_and_add_direction(captions, directions, prompt_pool):
    # Clean the captions by removing the unwanted prefixes
    captions = clean_caption_prefix(captions)

    cleaned_captions = []
    for i, caption in enumerate(captions):
        # Randomly select a unique prompt from the pool
        prompt = random.choice(prompt_pool)
        prompt_pool.remove(prompt)  # Avoid repeating prompts

        # Add the cleaned caption with a dynamic prompt and direction
        cleaned_caption = f"{prompt.format(direction=directions[i])} {caption}"

        # Append the cleaned caption
        cleaned_captions.append(cleaned_caption)

    return cleaned_captions


# Example of how you would process the file
def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    image_data = {}

    for line in lines:
        if line.strip():
            image_name, caption = line.split(": ")
            image_name = image_name.strip()

            image_name = image_name.split("_")[0]  # Extract base image name

            if image_name not in image_data:
                image_data[image_name] = []

            image_data[image_name].append(caption)

    # Now process each image's data
    merged_captions = {}

    for base_image_name, captions in image_data.items():
        if len(captions) == 4:
            # Copy the prompt_pool to avoid modifying the original one
            cleaned_captions = clean_and_add_direction(captions, directions, prompt_pool.copy())
            merged_captions[base_image_name] = " ".join(cleaned_captions)

    return merged_captions


# Function to save prompts to a file with image name and prompt
def save_prompts_to_file(prompts, file_path):
    with open(file_path, 'w') as file:
        for image_name, prompt in prompts.items():
            file.write(f"{image_name}.png: {prompt}\n")


# Example usage:
file_path = r"C:\temp\FDenoisedSplitted\Florence-captions.txt"  # Replace with the path to your text file
merged_captions = process_file(file_path)

# Save the prompts to a file located at C:\temp\FDenoised with image name and prompt
save_file_path = r"C:\temp\FDenoised\Merged-Florence-captions.txt"  # Specify the path where you want to save the prompts
save_prompts_to_file(merged_captions, save_file_path)

print(f"Prompts have been saved to: {save_file_path}")
