"""
Quick manual test for the ingestion pipeline (Stage 2).

Picks a sample PDF from data/samples/, extracts its text using the
digital-text path when possible, or falls back to OCR (with preprocessing)
for scanned/image-only PDFs.

If data/samples/ has no PDFs yet, this script generates one synthetic
scanned-style sample (an image with no text layer) so the OCR path can
be exercised end-to-end.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pdf_parser import (
    is_text_based_pdf,
    extract_text_from_pdf,
    extract_images_from_pdf,
)
from ingestion.ocr import ocr_pdf_pages
from ingestion.preprocess import preprocess_image

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "samples")


def find_sample_pdf() -> str:
    matches = glob.glob(os.path.join(SAMPLES_DIR, "**", "*.pdf"), recursive=True)
    if matches:
        return matches[0]
    return generate_synthetic_scanned_pdf()


def generate_synthetic_scanned_pdf() -> str:
    """
    Create a fake scanned medical report: text drawn onto an image and
    embedded into a PDF page with no real text layer, so is_text_based_pdf()
    returns False and the OCR path gets exercised.
    """
    import fitz
    from PIL import Image, ImageDraw, ImageFont

    out_dir = os.path.join(SAMPLES_DIR, "lab")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_scanned_sample.pdf")

    # PIL's default bitmap font is tiny and low-quality, which makes OCR
    # unrealistically bad. A real scanned document has legible print, so
    # use a proper TrueType font at a readable size to actually simulate one.
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except OSError:
        font = ImageFont.load_default()

    img = Image.new("RGB", (1700, 2200), color="white")
    draw = ImageDraw.Draw(img)
    lines = [
        "MEDICAL LABORATORY REPORT",
        "",
        "Patient: John Doe",
        "Date: 2026-08-26",
        "",
        "Test Name: Complete Blood Count (CBC)",
        "",
        "Hemoglobin: 13.5 g/dL (12.0-15.5 g/dL, Normal)",
        "WBC Count: 7200 /uL (4500-11000 /uL, Normal)",
        "Platelet Count: 250000 /uL (150000-450000 /uL, Normal)",
        "",
        "Impression: No abnormalities detected.",
    ]
    y = 100
    for line in lines:
        draw.text((100, y), line, fill="black", font=font)
        y += 80

    doc = fitz.open()
    page = doc.new_page(width=img.width, height=img.height)
    img_path = out_path.replace(".pdf", "_tmp.png")
    img.save(img_path)
    page.insert_image(page.rect, filename=img_path)
    doc.save(out_path)
    doc.close()
    os.remove(img_path)

    print(f"[setup] No PDFs found in data/samples/ - generated synthetic sample:\n  {out_path}\n")
    return out_path


def main():
    pdf_path = find_sample_pdf()
    print(f"Using sample PDF: {pdf_path}")

    if is_text_based_pdf(pdf_path):
        print("Detected: digital (text-based) PDF -> extracting text directly.\n")
        text = extract_text_from_pdf(pdf_path)
    else:
        print("Detected: scanned/image-based PDF -> running OCR pipeline.\n")
        images = extract_images_from_pdf(pdf_path)
        images = [preprocess_image(img) for img in images]
        text = ocr_pdf_pages(images)

    print("----- EXTRACTED TEXT -----")
    print(text.strip())
    print("---------------------------")


if __name__ == "__main__":
    main()
