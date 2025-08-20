DOMAINS = ["Private Equity", "Real Estate", "Fiscal/Tax", "Wealth Management"]

STYLE_GUIDE = (
    "Tone: formal, precise, concise. Audience: investors & regulatory.\n"
    "Constraints: preserve numbers, percentages, dates and legal entity names. "
    "Enforce preferred terms. Locale formats: en 1,234.56 | fr 1 234,56 | de 1.234,56 | es 1.234,56.\n"
    "Avoid calques; prefer established industry phrasing. Honor do-not-translate list."
)

def normalize_domain(label: str) -> str:
    label = (label or "").strip().lower()
    for d in DOMAINS:
        if d.lower() == label:
            return d
    if "equity" in label:
        return "Private Equity"
    if "estate" in label or "rics" in label or "cap rate" in label:
        return "Real Estate"
    if "tax" in label or "vat" in label or "withholding" in label or "fiscal" in label:
        return "Fiscal/Tax"
    return "Wealth Management"
