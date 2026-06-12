"""VLM stage — identify / filter / fold / compress objects before box packing.

Uses Google Gemini when `GEMINI_API_KEY` is set (model `GEMINI_MODEL`, default
`gemini-3.5-flash`); otherwise falls back to a deterministic heuristic. Either
way it returns one record per input object:

    {instance_id, keep, identity, material, compressible, foldable,
     corrected_dims_cm, reasoning, source}

`source` is "gemini" or "heuristic". The call never raises — any API/model error
degrades to the heuristic so the pipeline keeps running.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

from box_fit import best_fit_box

log = logging.getLogger(__name__)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

# Soft material keywords → compressible / foldable (used by both paths).
_FOLDABLE = ("curtain", "sheet", "blanket", "duvet", "comforter", "bedding",
             "towel", "clothes", "clothing", "garment", "fabric", "커튼", "이불", "옷")
_COMPRESSIBLE = _FOLDABLE + ("cushion", "pillow", "mattress", "foam", "베개", "쿠션", "매트")


# --------------------------------------------------------------------------- #
# Heuristic fallback
# --------------------------------------------------------------------------- #
def _heuristic(objects: list[dict], catalog=None) -> list[dict]:
    out = []
    for o in objects:
        dims = sorted(float(d) for d in o["dims_cm"])
        longest = dims[2]
        # Drop implausibly large blobs (merged scene / background): > 2 m a side.
        keep = longest <= 200.0
        box = ""
        if catalog:
            b, _ = best_fit_box(dims, catalog)
            box = b.name if b else ""
        out.append({
            "instance_id": o["instance_id"],
            "keep": bool(keep),
            "duplicate_of": -1,
            "identity": o.get("label") or "물체",
            "material": "미상",
            "compressible": False,
            "foldable": False,
            "corrected_dims_cm": [round(d, 1) for d in dims],
            "box": box,
            "reasoning": "휴리스틱(VLM 없음): 한 변이 200cm 초과가 아니면 유지; "
                         "박스는 기하학적으로 가장 작은 것",
            "source": "heuristic",
        })
    return out


# --------------------------------------------------------------------------- #
# Gemini path
# --------------------------------------------------------------------------- #
_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "instance_id": {"type": "integer"},
                    "keep": {"type": "boolean"},
                    "duplicate_of": {"type": "integer"},
                    "identity": {"type": "string"},
                    "material": {"type": "string"},
                    "compressible": {"type": "boolean"},
                    "foldable": {"type": "boolean"},
                    "corrected_dims_cm": {"type": "array", "items": {"type": "number"}},
                    "box": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["instance_id", "keep", "duplicate_of", "identity",
                             "material", "compressible", "foldable",
                             "corrected_dims_cm", "box", "reasoning"],
            },
        }
    },
    "required": ["objects"],
}

_PROMPT = """당신은 이삿짐 포장/적재 결정자입니다. 이미지에는 SAM 마스크 윤곽, 3D 박스, id가
표시되어 있습니다. SAM은 라벨이 없고 벽/바닥/그림자/반사 같은 오검출이나 한 물체의 중복 마스크를
만들 수 있습니다. 아래 측정값은 단일 시점 깊이라 두께가 작거나 마스크 번짐으로 일부 축이 클 수
있습니다. 이미지의 id와 아래 목록을 맞춰 각 id를 판단하세요.

반드시 JSON 객체 하나만 반환하세요. 최상위 키는 "objects"이고, 각 항목은 아래 필드를 모두 가집니다:
instance_id, keep, duplicate_of, identity, material, compressible, foldable, corrected_dims_cm, box, reasoning

판단 규칙:
1. keep: 실제로 옮길 가정용 물건이면 true. 벽/바닥/천장/문/창문/문틀/창틀/기둥/붙박이장 등
   건물 구조와 방 자체, 가느다란 조각/그림자/반사/흐린 덩어리는 false.
2. 벽과 직물 구분: 벽은 방의 경계인 단단하고 평평한 배경이며 주름, 천 질감, 자유 끝단이 없습니다.
   단지 평평하거나 단색이라는 이유로 벽이라고 하지 마세요. 주름/드레이프/천 질감/봉/레일/펄럭이는
   끝단이 보이면 커튼 또는 직물입니다. 옷더미는 울퉁불퉁한 입체 더미이고 평평한 벽면이 아닙니다.
   천조각, 직물조각, 드레이프 천, 커튼, 천은 모두 하나로 합쳐 만약 커튼이 있을 때는 커튼 취급, 옷이 아닌 천조각들만 잡힌 경우에는 커튼 취급, 그 외에는 옷더미 취급한다.
3. duplicate_of: 같은 물리적 물체의 중복/부분 마스크이면 가장 작은 같은 물체 id를 넣고, 아니면 -1.
   박스가 크게 겹치거나 같은 깊이거나 한 박스가 다른 박스 안에 있으면 중복 가능성을 확인하세요.
   닿아 있거나 쌓였어도 identity/material이 다르면 병합하지 마세요.
4. identity/material: 짧은 한국어 품목명과 재질을 쓰세요. 커튼은 얇고 걸린 천, 매트리스는 두껍고
   단단한 슬래브입니다. 러그/카펫은 바닥의 평면 직물, 이불/담요는 부드러운 침구입니다.
5. compressible: 침구, 옷, 쿠션, 커튼, 베개, 수건처럼 눌리는 부드러운 물건만 true. 가구, 가전,
   전자기기, 식기, 책, 조명, 박스처럼 단단한 물건은 false.
6. foldable: 커튼, 시트, 수건, 러그처럼 많이 접히는 평면 직물만 true. 
7. corrected_dims_cm: 실제 포장 후 크기 [w,h,d] cm. 축마다 다르게 보정하세요.
   - 단단한 물건: 측정 크기 유지.
   - 커튼/러그/카펫/매트 같은 한 장짜리 평면 직물: 길이 방향으로 단단히 말거나 접어서, 길이 한 축만
     남기고 나머지 두 축(폭·두께)은 아주 얇게 만드세요 1~9 cm면 충분합니다. 말아놓은 천처럼 단면이 가늘어야 합니다 —
     두툼한 덩어리나 얇고 넓은 판으로 두면 안 됩니다.
   - 이불/담요/침구: 진공/압축. 큰 두 축은 조금 줄이고 두께 축을 크게 줄이세요.
   - 옷더미/수건더미/천가방: 세 축을 적당히 줄이되, 부피가 있으므로 너무 얇은 띠로는 만들지 마세요.
   - 베개/쿠션: 면 크기는 유지하고 두께만 주로 줄이세요.
8. box: corrected_dims_cm가 들어가는 카탈로그의 가장 작은 박스 이름. 단단한 물건이 어떤 박스에도
   안 들어가면 ""로 두세요. 박스에 억지로 넣는 것은 눌리는 소프트 짐에만 허용합니다.
9. reasoning: 제외/병합/포장 판단 이유를 한국어 한 문장으로 쓰세요.

사용 가능한 박스 (이름: w x h x d cm):
{boxlist}

물체 (id = 이미지에 그려진 번호; 측정 w x h x d cm + 부피):
{objlist}

스키마에 맞는 JSON만 반환하세요."""


def _build_prompt(objects: list[dict], catalog=None) -> str:
    """Format the shared prompt with the object list + box catalog (both backends)."""
    def _line(o):
        d = sorted(float(x) for x in o["dims_cm"])
        liters = d[0] * d[1] * d[2] / 1000.0
        return (f"- id {o['instance_id']}: "
                f"{'x'.join(str(round(x, 1)) for x in d)} cm  (vol ~{liters:.0f} L)")
    objlist = "\n".join(_line(o) for o in objects)
    if catalog:
        boxlist = "\n".join(
            f"- {b.name}: {'x'.join(str(round(d, 1)) for d in b.sorted_cm)} cm"
            for b in catalog)
    else:
        boxlist = '(no box catalog provided — set box to "")'
    return _PROMPT.format(objlist=objlist, boxlist=boxlist)


def _merge_record(o: dict, r: dict, source: str, catalog=None) -> dict:
    """Build one output record from a model reply `r` for object `o`."""
    dims = sorted(float(d) for d in (r.get("corrected_dims_cm") or o["dims_cm"]))
    box = (r.get("box") or "").strip()
    if catalog and box and box not in {b.name for b in catalog}:
        box = ""                      # model named a box not in the catalog → drop
    try:
        dup = int(r.get("duplicate_of", -1))
    except (TypeError, ValueError):
        dup = -1
    return {
        "instance_id": o["instance_id"],
        "keep": bool(r.get("keep", True)),
        "duplicate_of": dup,
        "identity": r.get("identity", "물체"),
        "material": r.get("material", "미상"),
        "compressible": bool(r.get("compressible", False)),
        "foldable": bool(r.get("foldable", False)),
        "corrected_dims_cm": [round(d, 1) for d in dims],
        "box": box,
        "reasoning": r.get("reasoning", ""),
        "source": source,
    }


def _gemini(rgb: np.ndarray, objects: list[dict], catalog=None) -> list[dict] | None:
    try:
        from google import genai
        from google.genai import types
        from PIL import Image
    except Exception as e:  # noqa: BLE001
        log.warning("google-genai/PIL import failed: %s", e)
        return None

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None

    prompt = _build_prompt(objects, catalog)

    try:
        client = genai.Client(api_key=key)
        img = Image.fromarray(rgb.astype(np.uint8))
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_SCHEMA,
            ),
        )
        data = json.loads(resp.text)
        recs = {r["instance_id"]: r for r in data.get("objects", [])}
    except Exception as e:  # noqa: BLE001
        log.warning("Gemini call failed (%s); using heuristic.", e)
        return None

    out = []
    for o in objects:
        r = recs.get(o["instance_id"])
        if r is None:
            out.append(_heuristic([o], catalog)[0])
        else:
            out.append(_merge_record(o, r, "gemini", catalog))
    return out


def analyze(rgb: np.ndarray, objects: list[dict], catalog=None) -> list[dict]:
    """Return per-object VLM records. Never raises; falls back to heuristic."""
    if not objects:
        return []
    recs = _gemini(rgb, objects, catalog)
    if recs is not None:
        log.info("VLM via Gemini (%s)", GEMINI_MODEL)
        return recs
    return _heuristic(objects, catalog)
