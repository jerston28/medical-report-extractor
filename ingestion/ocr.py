"""
Handles OCR for scanned/image-based reports using Tesseract.
"""

import os
import shutil

import pytesseract
from PIL import Image

# If Tesseract is already on PATH, pytesseract finds it with no config
# needed. Otherwise, fall back to common install locations rather than
# hardcoding one machine's path, so this works for anyone else who
# clones the repo.
if shutil.which("tesseract") is None:
    _CANDIDATES = [
        os.path.expanduser(r"~\Downloads\tesseract.exe"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract",
    ]
    _found = next((p for p in _CANDIDATES if os.path.isfile(p)), None)
    if _found:
        pytesseract.pytesseract.tesseract_cmd = _found
    # If none found, leave pytesseract's default ("tesseract") in place -
    # ocr_image() will raise a clear TesseractNotFoundError when called,
    # rather than failing silently here.


def ocr_image(image: Image.Image) -> str:
    """
    Run OCR on a single PIL Image and return extracted text.
    """
    return pytesseract.image_to_string(image)


def ocr_pdf_pages(images: list) -> str:
    """
    Run OCR across multiple page images (from pdf_parser.extract_images_from_pdf)
    and combine into one text blob.
    """
    page_texts = [ocr_image(image) for image in images]
    return "\n---\n".join(page_texts)
