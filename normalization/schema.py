"""
Output schemas for normalized extraction results.

Stage 3's extractors return plain dicts/lists shaped however was
convenient to build with regex/NER. Stage 4 maps that into these fixed
dataclasses so every document of a given type - regardless of which
extractor produced it or how messy the source text was - comes out with
the same fields in the same shape.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LabResult:
    test_name: str
    value: float
    unit: Optional[str]
    reference_range: Optional[str]
    flag: Optional[str]
    canonical_test_name: str


@dataclass
class PrescriptionItem:
    drug_name: str
    canonical_drug_name: str
    dosage: Optional[str]
    frequency: Optional[str]
    duration: Optional[str]


@dataclass
class DischargeInfo:
    diagnosis: Optional[str]
    procedures: Optional[str]
    follow_up: Optional[str]
    canonical_diagnosis: Optional[str]
    diagnosis_entities: list = field(default_factory=list)
