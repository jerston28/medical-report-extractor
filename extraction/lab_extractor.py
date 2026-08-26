"""
Regex-based extraction of lab test results.

Targets lines of the form:
    "Hemoglobin: 13.5 g/dL (12.0-15.5 g/dL, Normal)"
i.e. "<test name>: <value> <unit> (<reference range>, <flag>)", where the
parenthesized part and the flag are both optional.
"""

import re

LAB_LINE_RE = re.compile(
    r"^(?P<test>[A-Za-z][A-Za-z0-9 /()%-]*?)\s*:\s*"
    r"(?P<value>\d+\.?\d*)\s*"
    r"(?P<unit>[A-Za-zµμ/%]+)?"
    r"(?:\s*\(\s*(?P<paren>[^)]+?)\s*\))?"
    r"\s*$",
    re.MULTILINE,
)

FLAG_WORDS = {"normal", "high", "low", "abnormal", "critical"}
RANGE_RE = re.compile(r"\d+\.?\d*\s*-\s*\d+\.?\d*")


def _parse_paren(paren: str):
    """
    A parenthesized suffix can hold a reference range, a flag word, or
    both separated by a comma (e.g. "12.0-15.5 g/dL, Normal"). Split it
    apart into (reference_range, flag).
    """
    if not paren:
        return None, None

    reference_range = None
    flag = None
    for part in paren.split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower() in FLAG_WORDS:
            flag = part.capitalize()
        elif RANGE_RE.search(part):
            reference_range = part
        elif flag is None:
            # Unrecognized text (e.g. just "Normal range") - keep as flag
            # so nothing silently gets dropped.
            flag = part
    return reference_range, flag


def extract_lab_values(text: str) -> list:
    """
    Extract lab results into a list of dicts:
    {"test_name", "value", "unit", "reference_range", "flag"}
    """
    results = []
    for match in LAB_LINE_RE.finditer(text):
        test_name = match.group("test").strip()
        value = float(match.group("value"))
        unit = (match.group("unit") or "").strip() or None
        reference_range, flag = _parse_paren(match.group("paren"))

        results.append({
            "test_name": test_name,
            "value": value,
            "unit": unit,
            "reference_range": reference_range,
            "flag": flag,
        })
    return results
