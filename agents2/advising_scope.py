"""Single-dataset scope boundary; no user-selectable campus configuration."""
import re


def mentions_campus(value):
    return isinstance(value, str) and bool(re.search(r"\bcampus\b|\bcampuses\b|\bnewark\b|\bcamden\b|\bnew[ -]brunswick\b", value, re.I))


def without_campus(value):
    if isinstance(value, dict):
        return {k: without_campus(v) for k, v in value.items() if "campus" not in k.lower()}
    if isinstance(value, list):
        return [without_campus(v) for v in value if not mentions_campus(v)]
    return value


def clean_parsed_scope(parsed):
    missing_before = parsed.get("missing_critical_info")
    for key in ("entities", "memory_updates", "missing_critical_info", "suggested_clarifications"):
        if key in parsed:
            parsed[key] = without_campus(parsed[key])
    if missing_before and not parsed.get("missing_critical_info"):
        parsed["needs_clarification"] = False
        if parsed.get("clarification_action") == "incomplete":
            parsed["clarification_action"] = "resume"
    if mentions_campus(parsed.get("reference_error")):
        parsed["reference_error"] = "Which specific course do you mean?"
