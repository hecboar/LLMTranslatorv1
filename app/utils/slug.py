# app/utils/slug.py
import re


def slugify(value: str) -> str:
value = value.strip().lower()
value = re.sub(r"[^a-z0-9]+", "_", value)
return re.sub(r"_+", "_", value).strip("_")