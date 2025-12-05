from typing import Any

def to_int(v: Any) -> int:
    return int(v)

def to_float(v: Any) -> float:
    return float(v)

def to_str(v: Any) -> str:
    return str(v)

def to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean: {v}")

def to_list(v: Any) -> list[str]:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        if v.strip() == "":
            return []
        return [x.strip() for x in v.split(",")]
    raise ValueError("Expected list or comma-separated string")