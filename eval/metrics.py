"""
Compares the pipeline's current extraction+normalization output against
hand-verified ground truth files, computing precision/recall/F1 for
field extraction accuracy.

Only ground truth files with "verified": true are used - unverified
files are just the pipeline's own output (see eval/annotate.py) and
scoring the pipeline against itself would trivially always be perfect.

Scoring approach: each record (a lab result, a prescription item, or the
single discharge dict) is broken down into (field_name, value) pairs.
For list-type records (lab/prescription), predicted records are matched
to ground truth records by an identifier field (canonical test/drug
name) before comparing their other fields - if a whole record is
missing or spurious, every one of its fields counts as a miss. This
gives a field-level precision/recall that also penalizes for
completely missed or hallucinated records, not just wrong values within
matched ones.
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

IDENTIFIER_FIELDS = {
    "lab": "canonical_test_name",
    "prescription": "canonical_drug_name",
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


def _record_pairs(record: dict) -> set:
    return {(k, str(v)) for k, v in record.items()}


def _compare_lists(gt_list: list, pred_list: list, id_field: str):
    gt_by_id = {r.get(id_field): r for r in gt_list}
    pred_by_id = {r.get(id_field): r for r in pred_list}

    tp = fp = fn = 0
    for identifier, gt_record in gt_by_id.items():
        gt_pairs = _record_pairs(gt_record)
        if identifier in pred_by_id:
            pred_pairs = _record_pairs(pred_by_id[identifier])
            tp += len(gt_pairs & pred_pairs)
            fn += len(gt_pairs - pred_pairs)
            fp += len(pred_pairs - gt_pairs)
        else:
            fn += len(gt_pairs)  # whole record missed entirely

    for identifier, pred_record in pred_by_id.items():
        if identifier not in gt_by_id:
            fp += len(_record_pairs(pred_record))  # whole record hallucinated

    return tp, fp, fn


def _compare_dict(gt_dict: dict, pred_dict: dict):
    gt_pairs = _record_pairs(gt_dict)
    pred_pairs = _record_pairs(pred_dict)
    tp = len(gt_pairs & pred_pairs)
    fn = len(gt_pairs - pred_pairs)
    fp = len(pred_pairs - gt_pairs)
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}


def compute_extraction_metrics(samples_dir: str = SAMPLES_DIR, ground_truth_dir: str = GROUND_TRUTH_DIR) -> dict:
    """
    Runs the pipeline on every sample that has a verified ground truth
    file and scores the output against it. Returns:
      {"per_sample": {rel_path: prf_dict, ...}, "overall": prf_dict}
    or, if no verified ground truth exists yet:
      {"per_sample": {}, "overall": None, "message": "..."}
    """
    gt_paths = sorted(glob.glob(os.path.join(ground_truth_dir, "**", "*.json"), recursive=True))

    per_sample = {}
    total_tp = total_fp = total_fn = 0
    evaluated_any = False

    for gt_path in gt_paths:
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

        if not ground_truth.get("verified"):
            continue

        rel_path = os.path.relpath(gt_path, ground_truth_dir)
        pdf_rel_path = os.path.splitext(rel_path)[0] + ".pdf"
        pdf_path = os.path.join(samples_dir, pdf_rel_path)

        if not os.path.exists(pdf_path):
            per_sample[rel_path] = {"error": f"Sample PDF not found: {pdf_rel_path}"}
            continue

        doc_type = ground_truth["doc_type"]
        text = get_raw_text(pdf_path)
        raw_extraction = EXTRACTORS[doc_type](text)
        predicted = to_jsonable(normalize_extraction(doc_type, raw_extraction))
        gt_fields = ground_truth["fields"]

        if doc_type in IDENTIFIER_FIELDS:
            tp, fp, fn = _compare_lists(gt_fields, predicted, IDENTIFIER_FIELDS[doc_type])
        else:
            tp, fp, fn = _compare_dict(gt_fields, predicted)

        per_sample[rel_path] = _prf(tp, fp, fn)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        evaluated_any = True

    if not evaluated_any:
        return {
            "per_sample": {},
            "overall": None,
            "message": (
                "No verified ground truth files found. Run eval/annotate.py, "
                "hand-correct the JSON files it creates in eval/ground_truth/, "
                "and set \"verified\": true on the ones you've reviewed."
            ),
        }

    return {"per_sample": per_sample, "overall": _prf(total_tp, total_fp, total_fn)}


if __name__ == "__main__":
    results = compute_extraction_metrics()

    if results["overall"] is None:
        print(results["message"])
    else:
        print("Per-sample results:")
        for rel_path, prf in results["per_sample"].items():
            print(f"  {rel_path}: {prf}")
        print("\nOverall:")
        print(f"  {results['overall']}")
