"""Moving-box catalog parsing and fit logic.

A box "fits" an object if every sorted dimension of the object is <= the
corresponding sorted box dimension (allowing free rotation, hence sorting both).
Among all boxes that fit, we pick the one with the smallest volume.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_DIM_RE = re.compile(r"(\d+)\s*[Xx]\s*(\d+)\s*[Xx]\s*(\d+)")

# Default moving-box catalog (cm, W×H×D order as printed on the box).
DEFAULT_CATALOG_DIMS: list[list[float]] = [
    [20, 70, 40],
    [37, 44, 40],
    [37, 44, 50],
    [150, 20, 64],
    [175, 15, 5],
]


def parse_dims(text: str) -> Optional[list[float]]:
    """Extract the first WxHxD triple (cm) found anywhere in `text`.

    Returns the three values in ORIGINAL order (not sorted). None if absent.
    """
    m = _DIM_RE.search(text)
    if not m:
        return None
    return [float(m.group(1)), float(m.group(2)), float(m.group(3))]


@dataclass
class Box:
    name: str
    dims_cm: list[float]          # original order
    sorted_cm: list[float]        # ascending

    @property
    def volume_cm3(self) -> float:
        return self.dims_cm[0] * self.dims_cm[1] * self.dims_cm[2]


def box_name_from_dims(dims: list[float]) -> str:
    return "box_" + "x".join(str(int(d)) for d in dims)


def default_box_catalog_items() -> list[dict]:
    """Default catalog as JSON-serializable [{name, dims_cm}, ...]."""
    return [{"name": box_name_from_dims(d), "dims_cm": d}
            for d in DEFAULT_CATALOG_DIMS]


def boxes_from_items(items) -> list[Box]:
    """Build Box objects from [{name, dims_cm:[w,h,d]}, ...]."""
    boxes: list[Box] = []
    for b in items or []:
        try:
            name = (str(b.get("name") or "").strip())
            dims = [float(x) for x in b.get("dims_cm", [])][:3]
        except (TypeError, ValueError):
            continue
        if len(dims) == 3 and all(d > 0 for d in dims):
            if not name:
                name = box_name_from_dims(dims)
            boxes.append(Box(name, dims, sorted(dims)))
    return boxes


def catalog_items_from_boxes(boxes: list[Box]) -> list[dict]:
    return [{"name": b.name, "dims_cm": [round(d, 1) for d in b.dims_cm]}
            for b in boxes]


def load_catalog(csv_path: str | Path) -> list[Box]:
    """Load box catalog CSV with columns: name, dims_cm (e.g. 37X44X40)."""
    boxes: list[Box] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dims = parse_dims(row["dims_cm"])
            if dims is None:
                continue
            boxes.append(Box(row["name"].strip(), dims, sorted(dims)))
    return boxes


def catalog_from_folders(folder_names: list[str]) -> list[Box]:
    """Build the REAL box catalog from the dimension triples in folder names.

    The capture folder names (e.g. ``001_150X20X64``, ``015_...(19X30X38)``)
    encode the actual available boxes. We collect every distinct WxHxD triple
    (deduplicated by sorted dims) and name each box ``box_<w>x<h>x<d>``.
    """
    seen: dict[tuple, Box] = {}
    for name in folder_names:
        dims = parse_dims(name)
        if dims is None:
            continue
        key = tuple(sorted(dims))
        if key not in seen:
            label = "box_" + "x".join(str(int(d)) for d in dims)
            seen[key] = Box(label, dims, sorted(dims))
    # Stable order: by volume ascending.
    return sorted(seen.values(), key=lambda b: b.volume_cm3)


def best_fit_box(
    obj_dims_cm: list[float],
    boxes: list[Box],
    tol_cm: float = 1.0,
    tol_frac: float = 0.03,
) -> tuple[Optional[Box], list[dict]]:
    """Return (smallest fitting box | None, per-box detail list).

    Fit rule: sorted(obj)[i] <= sorted(box)[i] + max(tol_cm, tol_frac*box_i).
    """
    obj = sorted(obj_dims_cm)
    details = []
    fitting: list[Box] = []
    for b in boxes:
        margins = [b.sorted_cm[i] - obj[i] for i in range(3)]
        tols = [max(tol_cm, tol_frac * b.sorted_cm[i]) for i in range(3)]
        fits = all(obj[i] <= b.sorted_cm[i] + tols[i] for i in range(3))
        details.append({
            "box": b.name,
            "box_sorted_cm": b.sorted_cm,
            "margins_cm": [round(m, 1) for m in margins],
            "fits": fits,
        })
        if fits:
            fitting.append(b)
    if not fitting:
        return None, details
    best = min(fitting, key=lambda b: b.volume_cm3)
    return best, details
