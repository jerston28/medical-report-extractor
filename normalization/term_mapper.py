"""
Term normalization helpers for Stage 4.

Real medical reports are inconsistent: the same unit, drug, or test gets
written a dozen different ways depending on who typed it up. These
functions map that variation down to one canonical spelling per concept,
so downstream code (and Stage 5's summarizer) can rely on consistent
values instead of handling every variant itself.
"""

import re

from dateutil import parser as date_parser

# ---------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------

UNIT_ALIASES = {
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "mcg": "mcg", "microgram": "mcg", "micrograms": "mcg", "ug": "mcg",
    "g": "g", "gram": "g", "grams": "g",
    "ml": "mL", "milliliter": "mL", "milliliters": "mL",
    "iu": "IU",
    "gdl": "g/dL", "gmdl": "g/dL",
    "mgdl": "mg/dL",
    "ul": "/uL",
}


def _unit_key(unit: str) -> str:
    key = unit.strip().lower().replace("µ", "u").replace("μ", "u")
    return re.sub(r"[^a-z]", "", key)


def normalize_units(value, unit):
    """
    Standardize unit notation (case/symbol variants of the same
    measurement - e.g. "MG" -> "mg", "gm/dl" -> "g/dL", "/uL" and "/µL"
    both -> "/uL"). Returns (value, canonical_unit).

    `value` is accepted and passed back unchanged because these are
    purely notational variants of the same magnitude. A production
    system that also needed to convert *between* unit families (e.g.
    mcg <-> mg) would rescale value here too.
    """
    if not unit:
        return value, unit
    if unit.strip() == "%":
        return value, "%"
    canonical = UNIT_ALIASES.get(_unit_key(unit))
    return value, canonical if canonical else unit.strip()


# ---------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------

def normalize_date(date_string):
    """
    Parse a date written in any common format (DD/MM/YYYY, MM-DD-YY,
    "12 Jan 2026", etc.) into ISO format (YYYY-MM-DD). Returns None if
    the string can't be parsed as a date at all.
    """
    if not date_string:
        return None
    try:
        parsed = date_parser.parse(date_string, fuzzy=True)
    except (ValueError, OverflowError, TypeError):
        return None
    return parsed.date().isoformat()


# ---------------------------------------------------------------------
# Drug names
# ---------------------------------------------------------------------

# Small hardcoded lookup for a resume project. A production system would
# query the RxNorm/RxNav API (https://rxnav.nlm.nih.gov) to resolve drug
# name variants and brand names to a canonical RxNorm concept instead.
DRUG_NAME_ALIASES = {
    "acetaminophen": "Acetaminophen",
    "paracetamol": "Acetaminophen",
    "tylenol": "Acetaminophen",
    "panadol": "Acetaminophen",
    "ibuprofen": "Ibuprofen",
    "advil": "Ibuprofen",
    "motrin": "Ibuprofen",
    "amoxicillin": "Amoxicillin",
    "amoxil": "Amoxicillin",
    "metformin": "Metformin",
    "glucophage": "Metformin",
    "atorvastatin": "Atorvastatin",
    "lipitor": "Atorvastatin",
    "omeprazole": "Omeprazole",
    "prilosec": "Omeprazole",
    "amlodipine": "Amlodipine",
    "norvasc": "Amlodipine",
    "azithromycin": "Azithromycin",
    "zithromax": "Azithromycin",
}


def normalize_drug_name(drug_name):
    """
    Map a drug name variant or brand name to its canonical (generic)
    name. Falls back to title-casing unrecognized names so output
    formatting stays consistent even without a lookup hit.
    """
    if not drug_name:
        return drug_name
    key = drug_name.strip().lower()
    return DRUG_NAME_ALIASES.get(key, drug_name.strip().title())


# ---------------------------------------------------------------------
# Lab test names
# ---------------------------------------------------------------------

LAB_TEST_NAME_ALIASES = {
    "hb": "Hemoglobin", "hgb": "Hemoglobin", "haemoglobin": "Hemoglobin", "hemoglobin": "Hemoglobin",
    "wbc": "White Blood Cell Count", "wbc count": "White Blood Cell Count",
    "white blood cell count": "White Blood Cell Count", "white blood cells": "White Blood Cell Count",
    "plt": "Platelet Count", "platelets": "Platelet Count", "platelet count": "Platelet Count",
    "glucose": "Glucose", "blood glucose": "Glucose", "blood sugar": "Glucose", "fbs": "Glucose (Fasting)",
    "rbc": "Red Blood Cell Count", "rbc count": "Red Blood Cell Count",
    "creatinine": "Creatinine", "cr": "Creatinine",
    "sodium": "Sodium", "na": "Sodium",
    "potassium": "Potassium", "k": "Potassium",
}


def normalize_lab_test_name(test_name):
    """
    Map a lab test name variant/abbreviation to one canonical name
    (e.g. "Hb", "Hgb", "Haemoglobin" -> "Hemoglobin").
    """
    if not test_name:
        return test_name
    key = test_name.strip().lower()
    return LAB_TEST_NAME_ALIASES.get(key, test_name.strip())


# ---------------------------------------------------------------------
# Diagnosis abbreviations (supports DischargeInfo.canonical_diagnosis)
# ---------------------------------------------------------------------

DIAGNOSIS_ABBREVIATIONS = {
    "htn": "Hypertension",
    "t2dm": "Type 2 Diabetes Mellitus",
    "dm": "Diabetes Mellitus",
    "cad": "Coronary Artery Disease",
    "mi": "Myocardial Infarction",
    "chf": "Congestive Heart Failure",
    "copd": "Chronic Obstructive Pulmonary Disease",
    "ckd": "Chronic Kidney Disease",
    "uti": "Urinary Tract Infection",
    "afib": "Atrial Fibrillation",
}


def normalize_diagnosis(diagnosis_text):
    """
    Expand common clinical abbreviations in a comma-separated diagnosis
    string (e.g. "HTN, T2DM" -> "Hypertension, Type 2 Diabetes
    Mellitus"). Terms with no known abbreviation are left as-is, so the
    result only differs from the input when a mapping actually applies.
    """
    if not diagnosis_text:
        return diagnosis_text
    terms = [t.strip() for t in diagnosis_text.split(",")]
    expanded = [DIAGNOSIS_ABBREVIATIONS.get(t.lower(), t) for t in terms if t]
    return ", ".join(expanded)
