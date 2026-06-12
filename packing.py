"""Greedy 3D bin-packing of measured objects into real moving boxes.

This is a visualisation-grade heuristic (not optimal): first-fit-decreasing over
boxes, with axis-aligned shelf/layer placement inside each box. Compressible /
foldable objects may shrink to fit. Objects that fit no box become `unplaced`
and the viewer lays their OBBs end-to-end.

Coordinate convention: each box has its own local frame with origin at the
min-corner; placed objects carry `size_cm` (their axis-aligned size after any
fold/compress) and `pos_cm` (min-corner position inside the box).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# How much a soft item may be squashed per axis when packing.
COMPRESS_FACTOR = 0.7      # compressible (bedding, clothes): to 70 % per side
FOLD_FACTOR = 0.5          # foldable (curtain, sheets): halve the longest side


@dataclass
class PackedItem:
    instance_id: int
    label: str
    size_cm: list          # [w, h, d] after fold/compress (sorted ascending)
    pos_cm: list           # min-corner inside the box
    folded: bool = False
    compressed: bool = False
    crammed: bool = False   # VLM forced it into a box it doesn't geometrically fit


@dataclass
class PackedBox:
    name: str
    dims_cm: list          # sorted ascending [w, h, d]
    items: list = field(default_factory=list)


def _effective_size(obj: dict) -> tuple[list, bool, bool]:
    """Effective packed size of an object.

    ``dims_cm`` is the VLM's corrected size, which ALREADY reflects any fold /
    compress (the VLM gives the realistic folded/compressed dims), so we use it
    as-is — re-applying the factors here would double-shrink it. The fold/compress
    flags are kept only for display tags. If the dims look uncorrected (a heuristic
    run that didn't set corrected dims) we fall back to applying the factors.
    """
    dims = sorted(float(d) for d in obj["dims_cm"])
    folded = bool(obj.get("foldable"))
    compressed = bool(obj.get("compressible"))
    if obj.get("_raw_dims"):              # opt-in: caller passed measured (uncorrected) dims
        if folded:
            dims = sorted([dims[0], dims[1], dims[2] * FOLD_FACTOR])
        if compressed:
            dims = sorted(d * COMPRESS_FACTOR for d in dims)
    return dims, folded, compressed


def _try_place(box: PackedBox, size: list):
    """Find a min-corner position for an axis-aligned `size` inside `box`.

    Simple shelf packer: scan in z (height) layers, within a layer scan rows in y,
    within a row advance in x. Returns pos [x,y,z] or None.
    """
    W, D, H = box.dims_cm[2], box.dims_cm[1], box.dims_cm[0]  # x=long, y=mid, z=short
    sx, sy, sz = size[2], size[1], size[0]
    if sx > W + 1e-6 or sy > D + 1e-6 or sz > H + 1e-6:
        return None
    # Candidate positions: stack against already-placed items (corner heuristic).
    cands = [(0.0, 0.0, 0.0)]
    for it in box.items:
        px, py, pz = it.pos_cm
        ix, iy, iz = it.size_cm[2], it.size_cm[1], it.size_cm[0]
        cands += [(px + ix, py, pz), (px, py + iy, pz), (px, py, pz + iz)]
    cands.sort(key=lambda c: (c[2], c[1], c[0]))      # lowest, then back, then left
    for (x, y, z) in cands:
        if x + sx > W + 1e-6 or y + sy > D + 1e-6 or z + sz > H + 1e-6:
            continue
        if not _collides(box, x, y, z, sx, sy, sz):
            return [x, y, z]
    return None


def _collides(box: PackedBox, x, y, z, sx, sy, sz) -> bool:
    for it in box.items:
        px, py, pz = it.pos_cm
        ix, iy, iz = it.size_cm[2], it.size_cm[1], it.size_cm[0]
        if (x < px + ix - 1e-6 and x + sx > px + 1e-6 and
                y < py + iy - 1e-6 and y + sy > py + 1e-6 and
                z < pz + iz - 1e-6 and z + sz > pz + 1e-6):
            return True
    return False


def pack(objects: list[dict], catalog: list, allow_open_boxes: bool = True) -> dict:
    """Pack `objects` into moving boxes, CONSOLIDATING many items per box.

    Each object: {instance_id, label, dims_cm, compressible, foldable, box}.
    The VLM's `box` field is used only to decide loose-vs-boxed, NOT to give every
    item its own tight box:
      - box == ""  → large furniture, rides loose on the truck (never boxed).
      - box == a name, or no `box` key → the item is boxed. ALL boxed items are
        bin-packed TOGETHER (first-fit-decreasing): each item goes into the first
        already-open box it fits, and a new box is opened only when no open box has
        room — so a roomful of small things shares a few boxes, not one box each.
    An item that fits NO catalog box: if the VLM explicitly chose a box for it, it is
    CRAMMED into the largest box (`crammed=True`, the VLM may over-stuff); otherwise
    (geometric/large furniture) it rides loose.

    `catalog`: list of Box (box_fit.Box). Returns {"boxes":[...], "unplaced":[...]}.
    """
    cat = sorted(catalog, key=lambda b: b.volume_cm3)        # small → large
    items = []
    for o in objects:
        size, folded, compressed = _effective_size(o)
        items.append((o, size, folded, compressed))
    items.sort(key=lambda t: -(t[1][0] * t[1][1] * t[1][2]))  # volume desc (FFD)

    boxes: list[PackedBox] = []
    unplaced: list[dict] = []

    def _loose(o, size, folded, compressed):
        unplaced.append({
            "instance_id": o["instance_id"], "label": o.get("label", ""),
            "size_cm": [round(s, 1) for s in size],
            "folded": folded, "compressed": compressed,
        })

    def _add(box, o, size, folded, compressed, pos, crammed=False):
        box.items.append(PackedItem(o["instance_id"], o.get("label", ""),
                                    size, pos, folded, compressed, crammed))

    for o, size, folded, compressed in items:
        raw = o.get("box")
        # pref: "" = VLM said load loose; a name = VLM chose that box; None = no
        # decision (missing or null) -> just bin-pack it.
        pref = raw.strip() if isinstance(raw, str) else None

        # VLM EXPLICITLY said "load loose" (large furniture) -> straight to the truck.
        if pref == "":
            _loose(o, size, folded, compressed)
            continue

        # consolidate: drop it into the first ALREADY-OPEN box that has room.
        placed = False
        for box in boxes:
            pos = _try_place(box, size)
            if pos is not None:
                _add(box, o, size, folded, compressed, pos)
                placed = True
                break
        if placed:
            continue

        # none had room -> open a NEW box. Pick the LARGEST box that fits this item
        # (most room for the items still to come) so things consolidate into a few
        # boxes instead of one tight box each.
        if allow_open_boxes:
            for b in sorted(cat, key=lambda b: -b.volume_cm3):   # large → small
                trial = PackedBox(b.name, list(b.sorted_cm))
                pos = _try_place(trial, size)
                if pos is not None:
                    _add(trial, o, size, folded, compressed, pos)
                    boxes.append(trial)
                    placed = True
                    break
        if placed:
            continue

        # fits no box at all. Cramming = squashing it in past its size — that ONLY
        # makes sense for SOFT items (bedding, clothes, curtains) that actually give.
        # A RIGID item (microwave, furniture, appliance) can never be crammed; it
        # rides LOOSE on the truck instead.
        if pref and cat and (compressed or folded):
            big = cat[-1]
            nb = PackedBox(big.name, list(big.sorted_cm))
            _add(nb, o, size, folded, compressed, [0.0, 0.0, 0.0], crammed=True)
            boxes.append(nb)
        else:
            _loose(o, size, folded, compressed)

    return {
        "boxes": [{
            "name": b.name,
            "dims_cm": [round(d, 1) for d in b.dims_cm],
            "items": [{
                "instance_id": it.instance_id, "label": it.label,
                "size_cm": [round(s, 1) for s in it.size_cm],
                "pos_cm": [round(p, 1) for p in it.pos_cm],
                "folded": it.folded, "compressed": it.compressed,
                "crammed": it.crammed,
            } for it in b.items],
        } for b in boxes],
        "unplaced": unplaced,
    }
