"""
Runs Stage 3 extractor output through term_mapper's normalization
functions and returns it in the Stage 4 schema (normalization/schema.py).
"""

from normalization.term_mapper import (
    normalize_units,
    normalize_drug_name,
    normalize_lab_test_name,
    normalize_diagnosis,
)
from normalization.schema import LabResult, PrescriptionItem, DischargeInfo


def normalize_lab_result(raw: dict) -> LabResult:
    value, unit = normalize_units(raw.get("value"), raw.get("unit"))
    return LabResult(
        test_name=raw.get("test_name"),
        value=value,
        unit=unit,
        reference_range=raw.get("reference_range"),
        flag=raw.get("flag"),
        canonical_test_name=normalize_lab_test_name(raw.get("test_name")),
    )


def normalize_lab_results(raw_results: list) -> list:
    return [normalize_lab_result(r) for r in raw_results]


def normalize_prescription_item(raw: dict) -> PrescriptionItem:
    return PrescriptionItem(
        drug_name=raw.get("drug"),
        canonical_drug_name=normalize_drug_name(raw.get("drug")),
        dosage=raw.get("dosage"),
        frequency=raw.get("frequency"),
        duration=raw.get("duration"),
    )


def normalize_prescription_items(raw_items: list) -> list:
    return [normalize_prescription_item(r) for r in raw_items]


def normalize_discharge_info(raw: dict) -> DischargeInfo:
    diagnosis = raw.get("diagnosis")
    return DischargeInfo(
        diagnosis=diagnosis,
        procedures=raw.get("procedures"),
        follow_up=raw.get("follow_up"),
        canonical_diagnosis=normalize_diagnosis(diagnosis) if diagnosis else diagnosis,
        diagnosis_entities=raw.get("diagnosis_entities") or [],
    )


NORMALIZERS = {
    "lab": normalize_lab_results,
    "prescription": normalize_prescription_items,
    "discharge": normalize_discharge_info,
}


def normalize_extraction(doc_type: str, raw_extraction):
    """
    Dispatch raw extractor output to the right normalizer based on
    document type, returning schema dataclass instance(s).
    """
    normalizer = NORMALIZERS.get(doc_type)
    if normalizer is None:
        raise ValueError(f"No normalizer for document type: {doc_type!r}")
    return normalizer(raw_extraction)
