"""Final bin-packing: load the packed boxes (+ loose furniture) onto a truck and
quote by tonnage.

The packing pipeline ends here. Stage 1 (``packing.py``) puts items into moving
boxes; this module takes those boxes plus any loose furniture that fits no box
and 3-D packs them into the smallest / cheapest Korean cargo truck(s), then
prices the move by truck tonnage.

Truck catalog = real Korean enclosed cargo trucks (탑차 / 윙바디), internal cargo
dimensions in cm, covering 1톤 up to the 25톤 class (20톤급+). Dimensions are the
usable interior of the box/wing body (the relevant constraint for a move), not
the open flat-bed gate height.  Sources: carcar.kr / sendy.ai / dabori spec tables.

Coordinate frame inside a truck: x = length (front→back), y = width, z = height,
origin at the front-bottom-left corner. Placed cargo carries `size_cm` [x,y,z]
and `pos_cm` [x,y,z] (min corner) — ready for the viewer with no remapping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

# Fraction of the raw interior volume we treat as realistically usable (aisles,
# irregular stacking). A geometric pack already leaves gaps; this is a guard so a
# truck isn't reported as fitting cargo it never could in practice.
PACK_EFFICIENCY = 0.88


@dataclass(frozen=True)
class Truck:
    name: str
    tonnage: float
    dims_cm: tuple          # (length, width, height) interior, cm
    price_krw: int          # all-in single-trip MOVE fare per truck (운임+인력), editable

    @property
    def volume_cm3(self) -> float:
        return self.dims_cm[0] * self.dims_cm[1] * self.dims_cm[2]

    @property
    def usable_cm3(self) -> float:
        return self.volume_cm3 * PACK_EFFICIENCY


# Real Korean enclosed cargo trucks (탑차/윙바디). Interior L×W×H (cm) + ballpark
# single-trip MOVE fare (KRW) — this is a real 이사 price, NOT bare vehicle hire:
# it bundles the truck + a driver + loading/unloading crew (인부 인건비) for one
# metro-area trip (반포장/일반이사 기준). Bigger trucks carry more crew, so the
# fare climbs with tonnage. Volume is the binding constraint for a household move.
DEFAULT_TRUCKS: list[Truck] = [
    Truck("1톤 탑차",        1.0,  (280, 160, 180),  350_000),
    Truck("1.4톤 탑차",      1.4,  (310, 175, 190),  430_000),
    Truck("2.5톤 탑차",      2.5,  (430, 180, 200),  650_000),
    Truck("3.5톤 윙바디",    3.5,  (500, 200, 210),  850_000),
    Truck("5톤 윙바디",      5.0,  (620, 230, 230),  1_200_000),
    Truck("5톤 축차(장축)",  5.0,  (720, 235, 235),  1_400_000),
    Truck("8톤 윙바디",      8.0,  (810, 235, 240),  1_850_000),
    Truck("11톤 윙바디",     11.0, (910, 235, 240),  2_400_000),
    Truck("14톤 윙바디",     14.0, (950, 235, 245),  2_900_000),
    Truck("18톤 윙바디",     18.0, (1010, 235, 245), 3_500_000),
    Truck("25톤 윙바디",     25.0, (1010, 240, 245), 4_500_000),
]


def default_catalog() -> list[Truck]:
    return list(DEFAULT_TRUCKS)


def _orientations(size):
    """Unique axis-aligned orientations of a cuboid (rotations only)."""
    seen = set()
    for p in permutations(size):
        if p not in seen:
            seen.add(p)
            yield list(p)


def _collides(placed, x, y, z, sx, sy, sz) -> bool:
    for it in placed:
        px, py, pz = it["pos_cm"]
        ix, iy, iz = it["size_cm"]
        if (x < px + ix - 1e-6 and x + sx > px + 1e-6 and
                y < py + iy - 1e-6 and y + sy > py + 1e-6 and
                z < pz + iz - 1e-6 and z + sz > pz + 1e-6):
            return True
    return False


SUPPORT_MIN = 0.80      # a stacked item's base must be this fraction supported


def _supported(placed, x, y, z, sx, sy) -> bool:
    """Is a base [x,x+sx]x[y,y+sy] at height z adequately held up?

    On the floor (z≈0) always yes. Elevated, the base must rest on the tops of
    items below it covering >= SUPPORT_MIN of its area — so a wide item can't
    perch on a narrow one (that's what topples in transit).
    """
    if z <= 1e-6:
        return True
    need = sx * sy
    covered = 0.0
    for it in placed:
        px, py, pz = it["pos_cm"]
        ix, iy, iz = it["size_cm"]
        if abs((pz + iz) - z) > 1.0:        # its top isn't at this height
            continue
        ox = max(0.0, min(x + sx, px + ix) - max(x, px))
        oy = max(0.0, min(y + sy, py + iy) - max(y, py))
        covered += ox * oy
    return covered >= SUPPORT_MIN * need - 1e-6


def _place_one(placed, dims, size):
    """Find a min-corner pos + orientation for `size` in a container `dims` (L,W,H).

    Corner/shelf heuristic: candidate corners against already-placed cargo, fill
    bottom→front→left first. Returns (pos, oriented_size) or (None, None).

    STABILITY rules so the load looks/behaves realistically in transit:
      - laid flat: smallest dimension as height (z), longest along truck length (x);
        standing on end only as a last resort.
      - stacked only where it is supported (>= SUPPORT_MIN of its base rests on the
        tops below) — no wide item perched on a narrow one.
    Lowest valid position wins, so everything stacks from the floor up.
    """
    L, W, H = dims
    cands = [(0.0, 0.0, 0.0)]
    for it in placed:
        px, py, pz = it["pos_cm"]
        ix, iy, iz = it["size_cm"]
        cands += [(px + ix, py, pz), (px, py + iy, pz), (px, py, pz + iz)]
    cands.sort(key=lambda c: (c[2], c[0], c[1]))   # lowest, then front, then left
    # prefer flat (low height z), then longest side along truck length (x)
    oriented = sorted(_orientations(size), key=lambda s: (s[2], -s[0], -s[1]))
    for size_o in oriented:
        sx, sy, sz = size_o
        if sx > L + 1e-6 or sy > W + 1e-6 or sz > H + 1e-6:
            continue
        for (x, y, z) in cands:
            if x + sx > L + 1e-6 or y + sy > W + 1e-6 or z + sz > H + 1e-6:
                continue
            if _collides(placed, x, y, z, sx, sy, sz):
                continue
            if not _supported(placed, x, y, z, sx, sy):
                continue
            return [round(x, 1), round(y, 1), round(z, 1)], \
                   [round(sx, 1), round(sy, 1), round(sz, 1)]
    return None, None


def _fill_trucks(units, truck: Truck):
    """Greedily load `units` into as many copies of `truck` as needed.

    Returns list-of-loads (each a list of placed-unit dicts), or None if a single
    unit can't fit even an empty truck of this size.
    """
    # Load wide/heavy first so big footprints take the floor and everything stacks
    # up from there: sort by base footprint (two largest dims) desc, then volume.
    def _key(u):
        d = sorted(u["size"])
        return (-(d[1] * d[2]), -(d[0] * d[1] * d[2]))
    remaining = sorted(units, key=_key)
    loads = []
    while remaining:
        placed, load, leftover = [], [], []
        used = PACK_EFFICIENCY * truck.volume_cm3
        vol = 0.0
        for u in remaining:
            uv = u["size"][0] * u["size"][1] * u["size"][2]
            pos, size = (None, None)
            if vol + uv <= used + 1e-6:
                pos, size = _place_one(placed, truck.dims_cm, u["size"])
            if pos is None:
                leftover.append(u)
                continue
            rec = {"id": u["id"], "label": u["label"], "kind": u.get("kind", "box"),
                   "pos_cm": pos, "size_cm": size}
            placed.append(rec)
            load.append(rec)
            vol += uv
        if not load:                      # a unit too big for this truck entirely
            return None
        loads.append(load)
        remaining = leftover
    return loads


def plan(cargo_units: list[dict], catalog: list[Truck] | None = None) -> dict:
    """Choose the cheapest truck plan that carries every cargo unit, and quote it.

    `cargo_units`: [{id, label, kind, size:[x,y,z] cm}, ...] — packed boxes + loose
    furniture. For each truck type we compute how many trips/trucks are needed
    (greedy fill) and the total fare; the lowest total fare wins (tie → fewer
    trucks → smaller tonnage). Up to the 25톤 class.

    Returns a viz/quote dict, or an `empty`/`oversize` marker when nothing/too big.
    """
    catalog = catalog or default_catalog()
    units = [u for u in cargo_units if u and u.get("size") and all(s > 0 for s in u["size"])]
    total_vol = sum(u["size"][0] * u["size"][1] * u["size"][2] for u in units)

    if not units:
        return {"empty": True, "trucks": [], "count": 0, "quote_krw": 0,
                "cargo_volume_m3": 0.0, "options": []}

    options = []
    for t in sorted(catalog, key=lambda t: t.volume_cm3):
        loads = _fill_trucks(units, t)
        if loads is None:
            continue
        count = len(loads)
        total = count * t.price_krw
        util = total_vol / (count * t.volume_cm3) if count else 0.0
        options.append({"truck": t, "loads": loads, "count": count,
                        "total_krw": total, "utilization": round(util, 3)})

    if not options:
        biggest = max(catalog, key=lambda t: t.volume_cm3)
        over = [u for u in units if max(u["size"]) > max(biggest.dims_cm) + 1e-6]
        return {"oversize": True, "trucks": [], "count": 0, "quote_krw": 0,
                "cargo_volume_m3": round(total_vol / 1e6, 2),
                "oversize_items": [u["label"] for u in over], "options": []}

    best = min(options, key=lambda o: (o["total_krw"], o["count"], o["truck"].tonnage))
    t = best["truck"]
    return {
        "truck": {"name": t.name, "tonnage": t.tonnage,
                  "dims_cm": list(t.dims_cm), "price_krw": t.price_krw},
        "count": best["count"],
        "trucks": [{"index": i, "items": load} for i, load in enumerate(best["loads"])],
        "utilization": best["utilization"],
        "cargo_volume_m3": round(total_vol / 1e6, 2),
        "quote_krw": best["total_krw"],
        # alternative tonnages for transparency (cheapest option per tonnage)
        "options": [{
            "name": o["truck"].name, "tonnage": o["truck"].tonnage,
            "count": o["count"], "total_krw": o["total_krw"],
            "utilization": o["utilization"],
        } for o in sorted(options, key=lambda o: (o["total_krw"], o["count"]))[:6]],
    }
