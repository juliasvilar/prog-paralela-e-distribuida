import numpy as np
from PIL import Image

def load_image_as_array(filepath):
    img = Image.open(filepath).convert('RGB')
    return np.array(img, dtype=np.float32) # float32 é melhor para cálculos na GPU

def save_array_as_image(array, filepath):
    array = np.clip(array, 0, 255).astype(np.uint8)
    img = Image.fromarray(array)
    img.save(filepath)