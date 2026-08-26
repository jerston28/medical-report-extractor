"""
Prompt templates for turning normalized structured data (Stage 4 output)
into a plain-language patient summary.

Prompts are built from the *normalized* data, not raw report text -
grounding the LLM in verified structured fields (rather than noisy OCR
text) keeps the summary honest instead of letting the model guess at
things the extraction pipeline never actually confirmed.
"""

import dataclasses

SYSTEM_PROMPT = (
    "You are a medical report assistant that explains lab reports, "
    "prescriptions, and discharge summaries in plain, non-technical "
    "English for a patient with no medical background. Only explain "
    "the data you are given - do not add new medical advice, diagnoses, "
    "or recommendations beyond what the report already states. If "
    "nothing is flagged or abnormal, say so plainly."
)


def _as_dict_list(normalized_data):
    if isinstance(normalized_data, list):
        return [dataclasses.asdict(item) for item in normalized_data]
    return [dataclasses.asdict(normalized_data)]


def _build_lab_prompt(rows: list) -> str:
    lines = []
    for row in rows:
        flag = row["flag"] or "not flagged"
        lines.append(
            f"- {row['canonical_test_name']}: {row['value']} {row['unit'] or ''} "
            f"(reference range: {row['reference_range'] or 'not provided'}, flag: {flag})"
        )
    results_block = "\n".join(lines)

    return (
        "Here are the lab test results from a patient's report:\n\n"
        f"{results_block}\n\n"
        "Please respond with:\n"
        "1. Summary: a 2-3 sentence plain-English summary of this lab report overall.\n"
        "2. Flagged Findings: a list of any abnormal/flagged values from above, "
        "each explained simply (what it means for a value to be high/low here). "
        "If nothing is flagged, say so.\n"
    )


def _build_prescription_prompt(rows: list) -> str:
    lines = []
    for row in rows:
        parts = [row["canonical_drug_name"], row["dosage"] or "dosage not specified"]
        if row["frequency"]:
            parts.append(row["frequency"])
        if row["duration"]:
            parts.append(row["duration"])
        lines.append(f"- {' | '.join(p for p in parts if p)}")
    items_block = "\n".join(lines)

    return (
        "Here are the medications from a patient's prescription:\n\n"
        f"{items_block}\n\n"
        "Please respond with:\n"
        "1. Summary: a 2-3 sentence plain-English summary of this prescription overall.\n"
        "2. Medications Explained: for each medication, a simple explanation of what "
        "it is generally used for and how to take it, based on the dosage/frequency/"
        "duration given above.\n"
        "3. Flagged Concerns: note anything about the instructions that seems unclear "
        "or worth double-checking with a pharmacist/doctor (e.g. missing duration). "
        "If nothing stands out, say so.\n"
    )


def _build_discharge_prompt(info: dict) -> str:
    return (
        "Here is a patient's discharge summary:\n\n"
        f"- Diagnosis: {info['canonical_diagnosis'] or 'not documented'}\n"
        f"- Procedures performed: {info['procedures'] or 'not documented'}\n"
        f"- Follow-up instructions: {info['follow_up'] or 'not documented'}\n\n"
        "Please respond with:\n"
        "1. Summary: a 2-3 sentence plain-English summary of this hospital stay and "
        "why the patient was there.\n"
        "2. Flagged Findings: a list of the diagnoses/conditions above, each explained "
        "simply in plain language. If a diagnosis is not documented, say so.\n"
        "3. Follow-up: restate the follow-up instructions in plain, easy-to-follow terms.\n"
    )


def build_summary_prompt(normalized_data, doc_type: str) -> str:
    """
    Build the user-turn prompt for summarize_report(). `normalized_data`
    is the Stage 4 normalized output for the given doc_type (a list of
    LabResult/PrescriptionItem dataclasses, or a single DischargeInfo).
    """
    if doc_type == "lab":
        return _build_lab_prompt(_as_dict_list(normalized_data))
    if doc_type == "prescription":
        return _build_prescription_prompt(_as_dict_list(normalized_data))
    if doc_type == "discharge":
        return _build_discharge_prompt(_as_dict_list(normalized_data)[0])
    raise ValueError(f"Unknown doc_type: {doc_type!r}")
