# TexLidar  

TexLidar is a deep-learning-based tool for extracting text and detecting objects from ambient images. It utilizes Microsoft's Florence-2 model to analyze images, process captions, and identify objects within different sections of an image.  

## Features  

- **Caption Generation**: Splits an image into four sections (left, front, right, back) and generates descriptive captions.  
- **Object Detection**: Detects objects in each section, adjusts ROIs to full-image coordinates, and overlays bounding boxes.  
- **Automatic Image Processing**: Handles grayscale and 16-bit images, converting them for optimal model compatibility.  

## Installation  

### Prerequisites  
- Python 3.8+  
- PyTorch  
- Transformers (Hugging Face)  
- NumPy  
- PIL (Pillow)  
- Matplotlib  

### Install Dependencies  
```bash
pip install torch torchvision transformers numpy pillow matplotlib
```

## Usage  

### 1. Caption Generation  
Run the `Caption.py` script to generate a caption for an image:  
```bash
python Caption.py --image_path path/to/image.png
```
This will split the image into sections, generate captions for each part, and merge them into a final caption.  

### 2. Object Detection  
Run the `Od.py` script to detect objects and overlay bounding boxes:  
```bash
python Od.py --image_path path/to/image.png
```
This will analyze the image sections, detect objects, and display the full image with bounding boxes.  

## Project Structure  

```
TexLidar/
│── Caption.py       # Generates captions for image sections  
│── Od.py            # Performs object detection and overlays results  
│── utils.py         # Utility functions for image processing  
│── README.md        # Project documentation  
│── requirements.txt # Dependency list  
```

## Contribution  

Contributions are welcome! If you'd like to improve TexLidar, please open an issue or submit a pull request.  

## License  

This project is licensed under the MIT License.  

