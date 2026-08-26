"""
Handles digital (text-based) PDF extraction using PyMuPDF (fitz).
If a PDF has selectable text, use this - it's faster and more accurate than OCR.
"""

import io

import fitz  # pymupdf
from PIL import Image

TEXT_LENGTH_THRESHOLD = 20


def is_text_based_pdf(pdf_path: str) -> bool:
    """
    Check if a PDF has extractable text (vs. being just scanned images).
    Hint: open the PDF, check the first page's text length.
    If it's very short/empty, it's probably a scanned image PDF.
    """
    doc = fitz.open(pdf_path)
    try:
        first_page_text = doc[0].get_text()
    finally:
        doc.close()
    return len(first_page_text.strip()) > TEXT_LENGTH_THRESHOLD


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a digital PDF, page by page.
    """
    doc = fitz.open(pdf_path)
    try:
        pages_text = [page.get_text() for page in doc.pages()]
    finally:
        doc.close()
    return "\n".join(pages_text)


def extract_images_from_pdf(pdf_path: str) -> list:
    """
    For scanned PDFs: extract each page as an image so OCR can process it.
    Hint: use page.get_pixmap() to render each page as an image,
    then save or return as PIL Image objects for ocr.py to use.
    """
    doc = fitz.open(pdf_path)
    images = []
    try:
        for page in doc.pages():
            pix = page.get_pixmap(dpi=300)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(image)
    finally:
        doc.close()
    return images
