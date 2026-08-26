"""
End-to-end test for Stage 3 (Information Extraction).

For every sample file in data/samples/:
  1. Get raw text via the Stage 2 ingestion pipeline (digital text or OCR).
  2. Classify the document type (lab / prescription / discharge).
  3. Run the matching extractor.
  4. Print the structured result as JSON.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pdf_parser import is_text_based_pdf, extract_text_from_pdf, extract_images_from_pdf
from ingestion.ocr import ocr_pdf_pages
from ingestion.preprocess import preprocess_image
from classification.doc_type_classifier import classify_document
from extraction.lab_extractor import extract_lab_values
from extraction.prescription_extractor import extract_prescription_info
from extraction.discharge_extractor import extract_discharge_info

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "samples")

EXTRACTORS = {
    "lab": extract_lab_values,
    "prescription": extract_prescription_info,
    "discharge": extract_discharge_info,
}


def get_raw_text(pdf_path: str) -> str:
    if is_text_based_pdf(pdf_path):
        return extract_text_from_pdf(pdf_path)
    images = extract_images_from_pdf(pdf_path)
    images = [preprocess_image(img) for img in images]
    return ocr_pdf_pages(images)


def main():
    pdf_paths = sorted(glob.glob(os.path.join(SAMPLES_DIR, "**", "*.pdf"), recursive=True))
    if not pdf_paths:
        print("No sample PDFs found in data/samples/. Run ingestion/test_ingestion.py first.")
        return

    for pdf_path in pdf_paths:
        rel_path = os.path.relpath(pdf_path, SAMPLES_DIR)
        print("=" * 70)
        print(f"Sample: {rel_path}")

        text = get_raw_text(pdf_path)
        doc_type = classify_document(text)
        print(f"Classified as: {doc_type}")

        extractor = EXTRACTORS[doc_type]
        structured = extractor(text)

        print("Structured output:")
        print(json.dumps(structured, indent=2))
        print()


if __name__ == "__main__":
    main()
