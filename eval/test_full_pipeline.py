"""
Comprehensive test of the full pipeline (ingestion -> classification ->
extraction -> normalization -> summarization), reusing the exact
app.run_pipeline() function the Streamlit app calls, so this tests real
end-user behavior rather than a parallel/duplicated code path.

Runs 7 checks and prints a pass/fail table. Content is validated, not
just "did it crash" - see each check_N_* function's docstring.

Known, accepted limitations (not bugs - see README.md "Future
improvements" for the honest reasoning):
  - classify_document() is a keyword heuristic, not a trained model.
  - Regex extractors are tuned to the phrasing patterns seen in
    data/samples/ and won't catch every real-world phrasing variant.
  - OCR accuracy depends on scan quality; single-character misreads are
    possible (see eval/ground_truth/lab/synthetic_scanned_sample.json).
"""

import dataclasses
import glob
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

# Make stdout resilient to unicode the LLM sometimes returns (e.g. narrow
# no-break spaces), regardless of the calling terminal's codepage.
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except AttributeError:
    pass

import fitz

from ingestion.pdf_parser import is_text_based_pdf
import streamlit_app as app  # reuses the app's exact run_pipeline / get_raw_text_from_pdf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "data", "samples")
GROUND_TRUTH_DIR = os.path.join(PROJECT_ROOT, "eval", "ground_truth")
EDGE_CASES_DIR = os.path.join(SAMPLES_DIR, "edge_cases")
TYPE_FOLDERS = ["lab", "prescription", "discharge"]


# ---------------------------------------------------------------------
# Fixtures: create edge-case files and a multi-page sample if missing
# ---------------------------------------------------------------------

def ensure_edge_case_files():
    """(a) a non-medical image, (b) a 0-byte file, (c) a corrupted PDF."""
    os.makedirs(EDGE_CASES_DIR, exist_ok=True)
    paths = {
        "image": os.path.join(EDGE_CASES_DIR, "non_medical_image.png"),
        "empty": os.path.join(EDGE_CASES_DIR, "empty_file.pdf"),
        "corrupted": os.path.join(EDGE_CASES_DIR, "corrupted.pdf"),
    }
    created = []

    if not os.path.exists(paths["image"]):
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (800, 300), color="white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except OSError:
            font = ImageFont.load_default()
        draw.text((30, 30), "GROCERY LIST", fill="black", font=font)
        draw.text((30, 90), "- Milk", fill="black", font=font)
        draw.text((30, 130), "- Eggs", fill="black", font=font)
        draw.text((30, 170), "- Bread", fill="black", font=font)
        img.save(paths["image"])
        created.append(paths["image"])

    if not os.path.exists(paths["empty"]):
        open(paths["empty"], "wb").close()
        created.append(paths["empty"])

    if not os.path.exists(paths["corrupted"]):
        with open(paths["corrupted"], "wb") as f:
            f.write(b"%PDF-1.4\nThis is not a valid PDF body.\x00\x01\x02\xff\xfe" * 20)
        created.append(paths["corrupted"])

    return paths, created


def ensure_multipage_sample():
    """Find an existing multi-page sample, or create a synthetic 2-page discharge summary."""
    for t in TYPE_FOLDERS:
        for path in glob.glob(os.path.join(SAMPLES_DIR, t, "*.pdf")):
            doc = fitz.open(path)
            page_count = doc.page_count
            doc.close()
            if page_count > 1:
                return [path], False

    out_path = os.path.join(SAMPLES_DIR, "discharge", "discharge_sample_multipage.pdf")
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), (
        "DISCHARGE SUMMARY\n\n"
        "Patient: Alex Kim\n"
        "Date of Admission: 2026-06-01\n"
        "Date of Discharge: 2026-06-05\n\n"
        "Diagnosis:\n"
        "Community-Acquired Pneumonia"
    ), fontsize=12)
    page2 = doc.new_page()
    page2.insert_text((72, 72), (
        "Procedures:\n"
        "Sputum culture and chest X-ray performed.\n\n"
        "Follow-up:\n"
        "Complete the full course of antibiotics.\n"
        "Return for a follow-up chest X-ray in 4 weeks."
    ), fontsize=12)
    doc.save(out_path)
    doc.close()
    return [out_path], True


def get_type_samples():
    return {t: sorted(glob.glob(os.path.join(SAMPLES_DIR, t, "*.pdf"))) for t in TYPE_FOLDERS}


def rel(path):
    return os.path.relpath(path, SAMPLES_DIR)


def load_pdf_bytes(path):
    with open(path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------
# Check 1: coverage across document types
# ---------------------------------------------------------------------

def check1_coverage(samples_by_type):
    """Every sample's detected doc_type must match the folder it lives in."""
    rows = []
    for expected_type, paths in samples_by_type.items():
        for path in paths:
            result = app.run_pipeline(load_pdf_bytes(path), ".pdf")
            if "error" in result:
                rows.append((rel(path), False, f"pipeline error: {result['error']}"))
                continue
            actual_type = result["doc_type"]
            passed = actual_type == expected_type
            rows.append((rel(path), passed, f"expected={expected_type}, actual={actual_type}"))
    return rows


# ---------------------------------------------------------------------
# Check 2: digital vs scanned handling (behaviorally verified, not just
# re-deriving the same predicate - actually counts whether OCR ran)
# ---------------------------------------------------------------------

def check2_digital_vs_scanned(samples_by_type):
    rows = []
    for paths in samples_by_type.values():
        for path in paths:
            expected_digital = is_text_based_pdf(path)
            calls = {"ocr": 0}
            orig_ocr_pdf_pages = app.ocr_pdf_pages

            def counting_ocr_pdf_pages(images, _orig=orig_ocr_pdf_pages):
                calls["ocr"] += 1
                return _orig(images)

            app.ocr_pdf_pages = counting_ocr_pdf_pages
            try:
                app.get_raw_text_from_pdf(path)
            finally:
                app.ocr_pdf_pages = orig_ocr_pdf_pages

            actual_used_ocr = calls["ocr"] > 0
            expected_used_ocr = not expected_digital
            passed = actual_used_ocr == expected_used_ocr
            path_taken = "OCR (scanned)" if actual_used_ocr else "digital text (fast path)"
            print(f"  {rel(path)}: took {path_taken}")
            rows.append((rel(path), passed, f"expected_ocr={expected_used_ocr}, actual_ocr={actual_used_ocr}"))
    return rows


# ---------------------------------------------------------------------
# Check 3: extracted fields completeness
# ---------------------------------------------------------------------

REQUIRED_FIELDS = {
    "lab": ["test_name", "value", "unit", "reference_range", "flag"],
    "prescription": ["drug_name", "dosage", "frequency"],
    "discharge": ["diagnosis", "procedures", "follow_up"],
}


def check3_field_completeness(samples_by_type):
    rows = []
    for doc_type, paths in samples_by_type.items():
        required = REQUIRED_FIELDS[doc_type]
        for path in paths:
            result = app.run_pipeline(load_pdf_bytes(path), ".pdf")
            if "error" in result:
                rows.append((rel(path), False, f"pipeline error: {result['error']}"))
                continue

            normalized = result["normalized"]
            if doc_type == "discharge":
                records = [dataclasses.asdict(normalized)]
            else:
                if not normalized:
                    rows.append((rel(path), False, f"no {doc_type} rows extracted"))
                    continue
                records = [dataclasses.asdict(r) for r in normalized]

            missing = []
            for i, record in enumerate(records):
                for field in required:
                    if record.get(field) in (None, ""):
                        missing.append(f"record{i}.{field}")

            passed = not missing
            detail = "all required fields present" if passed else f"missing: {missing}"
            rows.append((rel(path), passed, detail))
    return rows


# ---------------------------------------------------------------------
# Check 4: summary quality
# ---------------------------------------------------------------------

def check4_summary_quality(samples_by_type):
    rows = []
    for paths in samples_by_type.values():
        for path in paths:
            result = app.run_pipeline(load_pdf_bytes(path), ".pdf")
            if "error" in result:
                rows.append((rel(path), False, f"pipeline error: {result['error']}"))
                continue

            summary = result["summary"]
            word_count = len(summary.split())
            has_error_marker = ("Error" in summary) or ("Traceback" in summary)
            passed = bool(summary.strip()) and word_count > 20 and not has_error_marker

            print(f"\n--- Summary for {rel(path)} ({word_count} words) ---")
            print(summary)
            print("---")

            rows.append((rel(path), passed, f"words={word_count}, error_markers={has_error_marker}"))
    return rows


# ---------------------------------------------------------------------
# Check 5: flagged/abnormal detection (lab samples, against ground truth)
# ---------------------------------------------------------------------

def _flagged_set(records):
    return {r["canonical_test_name"] for r in records if r.get("flag") and r["flag"].strip().lower() not in ("normal", "")}


def _normal_set(records):
    return {r["canonical_test_name"] for r in records if r.get("flag") and r["flag"].strip().lower() == "normal"}


def check5_flagged_detection():
    rows = []
    gt_paths = sorted(glob.glob(os.path.join(GROUND_TRUTH_DIR, "lab", "*.json")))
    for gt_path in gt_paths:
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)

        name = os.path.relpath(gt_path, GROUND_TRUTH_DIR)
        if not gt.get("verified"):
            rows.append((name, None, "skipped - ground truth not verified"))
            continue

        pdf_rel = os.path.splitext(os.path.relpath(gt_path, GROUND_TRUTH_DIR))[0] + ".pdf"
        pdf_path = os.path.join(SAMPLES_DIR, pdf_rel)

        expected_flagged = _flagged_set(gt["fields"])
        expected_normal = _normal_set(gt["fields"])

        result = app.run_pipeline(load_pdf_bytes(pdf_path), ".pdf")
        if "error" in result:
            rows.append((pdf_rel, False, f"pipeline error: {result['error']}"))
            continue

        actual_records = [dataclasses.asdict(r) for r in result["normalized"]]
        actual_flagged = _flagged_set(actual_records)

        missed = expected_flagged - actual_flagged
        false_flags = expected_normal & actual_flagged
        passed = not missed and not false_flags

        detail = f"expected_flagged={expected_flagged or set()}, actual_flagged={actual_flagged or set()}"
        if missed:
            detail += f", MISSED={missed}"
        if false_flags:
            detail += f", FALSE_FLAGS={false_flags}"
        rows.append((pdf_rel, passed, detail))
    return rows


# ---------------------------------------------------------------------
# Check 6: error handling / graceful failure
# ---------------------------------------------------------------------

def check6_error_handling(edge_case_paths):
    rows = []
    cases = [
        ("non_medical_image.png", edge_case_paths["image"], ".png"),
        ("empty_file.pdf", edge_case_paths["empty"], ".pdf"),
        ("corrupted.pdf", edge_case_paths["corrupted"], ".pdf"),
    ]
    for name, path, ext in cases:
        file_bytes = load_pdf_bytes(path)
        try:
            result = app.run_pipeline(file_bytes, ext)
            crash_detail = None
        except Exception as e:
            result = None
            crash_detail = repr(e)

        if crash_detail is not None:
            rows.append((name, False, f"UNHANDLED EXCEPTION: {crash_detail}"))
            continue

        graceful = isinstance(result, dict)
        if "error" in result:
            detail = f"returned graceful error message: {result['error']!r}"
        else:
            detail = "did not crash; produced a result (possibly empty fields) instead of an error"
        rows.append((name, graceful, detail))
    return rows


# ---------------------------------------------------------------------
# Check 7: multi-page handling
# ---------------------------------------------------------------------

def check7_multipage(multipage_paths):
    rows = []
    for path in multipage_paths:
        doc = fitz.open(path)
        page_count = doc.page_count
        doc.close()

        result = app.run_pipeline(load_pdf_bytes(path), ".pdf")
        if "error" in result:
            rows.append((rel(path), False, f"pipeline error: {result['error']}"))
            continue

        doc_type = result["doc_type"]
        normalized = result["normalized"]

        if doc_type == "discharge":
            d = dataclasses.asdict(normalized)
            page2_ok = bool(d.get("procedures")) and bool(d.get("follow_up"))
            passed = page_count > 1 and page2_ok
            detail = f"pages={page_count}, procedures={'present' if d.get('procedures') else 'MISSING'}, follow_up={'present' if d.get('follow_up') else 'MISSING'}"
        else:
            passed = page_count > 1 and len(result["raw_text"]) > 0
            detail = f"pages={page_count}, raw_text_len={len(result['raw_text'])}"

        rows.append((rel(path), passed, detail))
    return rows


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

CHECK_NAMES = {
    1: "1. Coverage across document types",
    2: "2. Digital vs scanned handling",
    3: "3. Extracted fields completeness",
    4: "4. Summary quality",
    5: "5. Flagged/abnormal detection",
    6: "6. Error handling / graceful failure",
    7: "7. Multi-page handling",
}


def print_summary_table(all_results):
    print("\n" + "=" * 78)
    print("FINAL PASS/FAIL SUMMARY")
    print("=" * 78)

    overall_pass = True
    for check_num in range(1, 8):
        rows = all_results[check_num]
        applicable = [r for r in rows if r[1] is not None]
        passed_count = sum(1 for r in applicable if r[1])
        total = len(applicable)
        status = "PASS" if total > 0 and passed_count == total else ("FAIL" if total > 0 else "N/A")
        if status == "FAIL":
            overall_pass = False

        print(f"\n{CHECK_NAMES[check_num]}: {passed_count}/{total} passed [{status}]")
        for name, passed, detail in rows:
            marker = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
            print(f"    [{marker}] {name}: {detail}")

    print("\n" + "=" * 78)
    print("OVERALL:", "ALL CHECKS PASSED" if overall_pass else "SOME CHECKS FAILED - see [FAIL] rows above")
    print("=" * 78)
    return overall_pass


def main():
    edge_case_paths, created_edge = ensure_edge_case_files()
    if created_edge:
        print("Created edge case files:")
        for p in created_edge:
            print(f"  - {rel(p)}")

    multipage_paths, created_multipage = ensure_multipage_sample()
    if created_multipage:
        print(f"No multi-page sample found - created: {rel(multipage_paths[0])}")

    samples_by_type = get_type_samples()

    all_results = {}

    print("\n--- CHECK 1: Coverage across document types ---")
    all_results[1] = check1_coverage(samples_by_type)

    print("\n--- CHECK 2: Digital vs scanned handling ---")
    all_results[2] = check2_digital_vs_scanned(samples_by_type)

    print("\n--- CHECK 3: Extracted fields completeness ---")
    all_results[3] = check3_field_completeness(samples_by_type)

    print("\n--- CHECK 4: Summary quality ---")
    all_results[4] = check4_summary_quality(samples_by_type)

    print("\n--- CHECK 5: Flagged/abnormal detection (lab samples) ---")
    all_results[5] = check5_flagged_detection()

    print("\n--- CHECK 6: Error handling / graceful failure ---")
    all_results[6] = check6_error_handling(edge_case_paths)

    print("\n--- CHECK 7: Multi-page handling ---")
    all_results[7] = check7_multipage(multipage_paths)

    overall_pass = print_summary_table(all_results)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
