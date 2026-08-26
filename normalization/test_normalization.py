"""
End-to-end test for Stage 4 (Normalization).

For every sample file in data/samples/, runs the full pipeline built so
far:
  ingestion (Stage 2) -> classification + extraction (Stage 3) ->
  normalization (Stage 4)

and prints a before/after comparison plus the final normalized JSON.
"""

import dataclasses
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
from normalization.normalize import normalize_extraction

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


def to_jsonable(normalized):
    if isinstance(normalized, list):
        return [dataclasses.asdict(item) for item in normalized]
    return dataclasses.asdict(normalized)


def print_before_after(doc_type: str, raw, normalized):
    print("Before/after (one field):")
    if doc_type == "lab":
        # Print every row where something actually changed, so the demo
        # stays meaningful even when most rows were already canonical.
        any_change = False
        for r, n in zip(raw, normalized):
            if r["test_name"] != n.canonical_test_name:
                print(f"  test_name : {r['test_name']!r} -> {n.canonical_test_name!r}")
                any_change = True
            if r["unit"] != n.unit:
                print(f"  unit      : {r['unit']!r} -> {n.unit!r}")
                any_change = True
        if not any_change:
            print("  (no changes - all values were already in canonical form)")
    elif doc_type == "prescription":
        any_change = False
        for r, n in zip(raw, normalized):
            if r["drug"] != n.canonical_drug_name:
                print(f"  drug      : {r['drug']!r} -> {n.canonical_drug_name!r}")
                any_change = True
        if not any_change:
            print(f"  drug      : {raw[0]['drug']!r} -> {normalized[0].canonical_drug_name!r} (no change)")
    elif doc_type == "discharge":
        print(f"  diagnosis : {raw['diagnosis']!r} -> {normalized.canonical_diagnosis!r}")
    print()


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

        raw_extraction = EXTRACTORS[doc_type](text)
        normalized = normalize_extraction(doc_type, raw_extraction)

        print_before_after(doc_type, raw_extraction, normalized)

        print("Normalized structured output:")
        print(json.dumps(to_jsonable(normalized), indent=2))
        print()


if __name__ == "__main__":
    main()
