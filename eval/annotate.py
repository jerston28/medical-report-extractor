"""
Generates ground-truth annotation templates for eval/ground_truth/.

For every sample PDF under data/samples/ that doesn't already have a
ground truth file, this runs the current pipeline (ingestion ->
classification -> extraction -> normalization) and writes its output as
a starting-point JSON file, marked "verified": false.

Why pre-fill from the pipeline instead of a blank template: hand-typing
every field for every sample from scratch is tedious and error-prone.
Correcting a draft is faster and less error-prone than writing one from
nothing - this is standard practice for building small eval sets. The
"verified" flag exists specifically so metrics.py can tell "reviewed by
a human" apart from "just what the pipeline happened to output" - only
verified files are used for scoring.

Never overwrites an existing ground truth file, so re-running this after
you've started hand-correcting won't lose your edits.
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "data", "samples")
GROUND_TRUTH_DIR = os.path.join(PROJECT_ROOT, "eval", "ground_truth")

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


def build_template(pdf_path: str) -> dict:
    text = get_raw_text(pdf_path)
    doc_type = classify_document(text)
    raw_extraction = EXTRACTORS[doc_type](text)
    normalized = normalize_extraction(doc_type, raw_extraction)
    return {
        "doc_type": doc_type,
        "verified": False,
        "fields": to_jsonable(normalized),
    }


def main():
    pdf_paths = glob.glob(os.path.join(SAMPLES_DIR, "**", "*.pdf"), recursive=True)
    # edge_cases/ holds deliberately broken files for eval/test_full_pipeline.py's
    # error-handling check - annotating "ground truth" for a corrupted file
    # makes no sense, so skip them.
    pdf_paths = sorted(p for p in pdf_paths if "edge_cases" not in os.path.relpath(p, SAMPLES_DIR).split(os.sep))
    if not pdf_paths:
        print("No sample PDFs found in data/samples/.")
        return

    created, skipped = [], []
    for pdf_path in pdf_paths:
        rel_path = os.path.relpath(pdf_path, SAMPLES_DIR)
        out_path = os.path.join(GROUND_TRUTH_DIR, os.path.splitext(rel_path)[0] + ".json")

        if os.path.exists(out_path):
            skipped.append(rel_path)
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        template = build_template(pdf_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2)
        created.append(rel_path)

    if created:
        print("Created ground truth templates for:")
        for rel_path in created:
            print(f"  - {rel_path}")
    if skipped:
        print("\nAlready exist (left untouched):")
        for rel_path in skipped:
            print(f"  - {rel_path}")

    print(
        "\nNext step: open the JSON files in eval/ground_truth/, correct any "
        "wrong field values by hand, and set \"verified\": true. Only "
        "verified files are used by eval/metrics.py."
    )


if __name__ == "__main__":
    main()
