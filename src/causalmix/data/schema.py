from __future__ import annotations

from dataclasses import dataclass
from typing import List

@dataclass
class DataSchema:
    """Column schema for real/synthetic evaluation."""
    numeric: List[str]
    binary: List[str]
    categorical: List[str]
