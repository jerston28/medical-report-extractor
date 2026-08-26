"""
Image preprocessing to improve OCR accuracy on messy scans.
Common steps: grayscale, denoise, threshold (binarize), deskew.
"""

import cv2
import numpy as np
from PIL import Image


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Clean up an image before OCR: grayscale + denoise + threshold.
    Hint: convert PIL Image to a numpy array/cv2 format, apply:
      1. cv2.cvtColor(...) to grayscale
      2. cv2.fastNlMeansDenoising(...) to reduce noise
      3. cv2.threshold(...) with cv2.THRESH_BINARY + cv2.THRESH_OTSU
    Then convert back to PIL Image before returning.
    """
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, binarized = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binarized)
