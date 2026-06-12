"""견적서 자동 예상.

파이프라인이 끝난 뒤, VLM(Gemini 또는 로컬 Gemma)으로 결과(짐 목록 + 배정 트럭 + 운임)를
훑어 견적서의 나머지 항목(작업유형·인원·비용·요청사항)을 합리적으로 예상한다. 사람은 UI에서
필요한 부분만 수정한다. 키/모델이 없으면 톤수·부피 기반 휴리스틱으로 대체한다 — 절대 raise하지
않는다.

반환 dict 키(만원 단위 비용은 숫자/문자):
  work_in_type, work_out_type, crew_in, crew_out,
  cost_in, cost_out, deposit, total, requests(list[str])
고객명·연락처·주소·일자 등 개인정보는 예상하지 않는다(공란 유지).
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)

_FIELDS = ("work_in_type", "work_out_type", "crew_in", "crew_out",
           "cost_in", "cost_out", "deposit", "total", "requests")

_SCHEMA = {
    "type": "object",
    "properties": {
        "work_in_type": {"type": "string"},
        "work_out_type": {"type": "string"},
        "crew_in": {"type": "string"},
        "crew_out": {"type": "string"},
        "cost_in": {"type": "number"},
        "cost_out": {"type": "number"},
        "deposit": {"type": "number"},
        "requests": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["work_in_type", "work_out_type", "crew_in", "crew_out",
                 "cost_in", "cost_out", "deposit", "requests"],
}

_PROMPT = """당신은 이사 견적 담당자입니다. 아래는 한 집의 이삿짐 인벤토리와 배정된 운반 트럭,
그리고 차량+인력 포함 예상 운임입니다. 이 정보를 바탕으로 견적서의 나머지 항목을 합리적으로
예상해 JSON으로 채우세요. 사람이 검토 후 수정합니다. 비용은 만원 단위 숫자입니다.

트럭: {truck}
예상 운임 합계(차량+인력): {total}만원
총 화물 부피: {vol} m3
짐 목록(품목 / 크기 / 배정 박스):
{items}

다음 JSON을 반환하세요:
- work_in_type: 출발지(입고) 작업유형 추정. 예: "1층작업", "엘리베이터 작업", "계단작업",
  "사다리차 작업". 단서가 없으면 "1층작업".
- work_out_type: 도착지(출고) 작업유형 추정(같은 보기 중에서).
- crew_in: 입고 작업 인원(남:녀). 예: "2:1". 짐 양과 트럭 톤수에 맞게.
- crew_out: 출고 작업 인원(남:녀).
- cost_in: 입고 총액(만원). 입고+출고 합이 위 예상 운임 합계와 비슷하도록.
- cost_out: 출고 총액(만원).
- deposit: 계약금(만원). 보통 전체의 10% 안팎.
- requests: 적절한 고객 요청사항(한국어 항목 배열). 예: 매트리스가 있으면 "매트리스 클리닉(서비스)",
  큰 가구가 많으면 "사다리차 필요", 가전이 있으면 "가전 분리/설치" 등. 없으면 빈 배열.
폐기짐은 추정하지 마세요 — 폐기 여부는 고객만 정합니다. 감지된 짐은 모두 옮기는 짐으로 간주하세요.
JSON만 반환하세요."""


# --------------------------------------------------------------------------- #
def _summarize(viz: dict) -> tuple[str, str, float, list[str]]:
    tp = viz.get("truck_plan") or {}
    truck = tp.get("truck") or {}
    count = tp.get("count", 1) or 1
    tlabel = f"{truck.get('name', '미정')} x {count}대" if truck else "미정"
    total = round((tp.get("quote_krw") or 0) / 10000)
    vol = tp.get("cargo_volume_m3", 0.0)
    items = []
    for o in viz.get("instances", []):
        if not o.get("kept", True):
            continue
        name = o.get("identity") or o.get("label") or f"물체 {o.get('id')}"
        dims = o.get("corrected_dims_cm") or o.get("dims_cm") or []
        size = "x".join(str(round(float(d))) for d in dims) + "cm" if dims else ""
        box = (o.get("box") or "").strip()
        items.append(f"- {name} / {size} / {box or 'loose 가구'}")
    return tlabel, total, vol, items


def _prompt(viz: dict) -> str:
    tlabel, total, vol, items = _summarize(viz)
    return _PROMPT.format(truck=tlabel, total=total, vol=vol,
                          items="\n".join(items) or "(짐 없음)")


# --------------------------------------------------------------------------- #
# heuristic fallback (no LLM)
# --------------------------------------------------------------------------- #
def _heuristic(viz: dict) -> dict:
    tp = viz.get("truck_plan") or {}
    truck = tp.get("truck") or {}
    tonnage = float(truck.get("tonnage", 1.0) or 1.0)
    total = round((tp.get("quote_krw") or 0) / 10000)

    if tonnage <= 1.4:
        crew = "2:0"
    elif tonnage <= 3.5:
        crew = "2:1"
    elif tonnage <= 5.0:
        crew = "3:1"
    elif tonnage <= 8.0:
        crew = "4:1"
    else:
        crew = "5:1"

    cost_in = round(total * 0.45)
    cost_out = total - cost_in            # 입고+출고 = 운임 합계
    deposit = max(1, round(total * 0.1))

    reqs = []
    names = " ".join((o.get("identity") or "") for o in viz.get("instances", [])
                     if o.get("kept", True))
    if "매트리스" in names:
        reqs.append("매트리스 클리닉(서비스)")
    if any(k in names for k in ("냉장고", "세탁기", "건조기", "에어컨", "TV", "티비")):
        reqs.append("가전 분리/설치")
    if any((o.get("box") or "").strip() == "" and o.get("kept", True)
           for o in viz.get("instances", [])):
        reqs.append("대형 가구 사다리차 필요 여부 확인")

    return {
        "work_in_type": "1층작업",
        "work_out_type": "1층작업",
        "crew_in": crew,
        "crew_out": crew,
        "cost_in": cost_in,
        "cost_out": cost_out,
        "deposit": deposit,
        "total": cost_in + cost_out + deposit,   # 견적서 관행: 합계 = 입고+출고+계약금
        "requests": reqs,
        "source": "heuristic",
    }


# --------------------------------------------------------------------------- #
# LLM paths
# --------------------------------------------------------------------------- #
def _normalize(rec: dict, viz: dict) -> dict:
    """Coerce an LLM record to the output dict, computing total per the form rule."""
    out = {}
    for k in ("work_in_type", "work_out_type", "crew_in", "crew_out"):
        v = rec.get(k)
        if v:
            out[k] = str(v)
    for k in ("cost_in", "cost_out", "deposit"):
        try:
            out[k] = round(float(rec[k]))
        except (KeyError, TypeError, ValueError):
            pass
    reqs = rec.get("requests")
    if isinstance(reqs, list):
        out["requests"] = [str(x) for x in reqs if str(x).strip()]
    if all(k in out for k in ("cost_in", "cost_out", "deposit")):
        out["total"] = out["cost_in"] + out["cost_out"] + out["deposit"]
    return out


def _gemini(viz: dict) -> dict | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
            contents=[_prompt(viz)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=_SCHEMA),
        )
        rec = json.loads(resp.text)
    except Exception as e:  # noqa: BLE001
        log.warning("견적 예상 Gemini 실패(%s); 휴리스틱 사용.", e)
        return None
    out = _normalize(rec, viz)
    out["source"] = "gemini"
    return out


def _gemma(viz: dict) -> dict | None:
    try:
        import config
        import gemma_vlm
    except Exception:  # noqa: BLE001
        return None
    backend = config.get("vlm_backend", "auto")
    path = gemma_vlm.model_path_for(backend if backend.startswith("gemma") else "auto")
    if not path:
        return None
    try:
        pipe = gemma_vlm._get_pipe(path)
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": _prompt(viz)}]}]
        out_raw = pipe(text=messages, max_new_tokens=512, do_sample=False)
        reply = gemma_vlm._reply_text(out_raw)
        rec = gemma_vlm._extract_json(reply)
        if isinstance(rec, list):
            rec = rec[0] if rec else {}
    except Exception as e:  # noqa: BLE001
        log.warning("견적 예상 Gemma 실패(%s); 휴리스틱 사용.", e)
        return None
    out = _normalize(rec, viz)
    out["source"] = "gemma"
    return out


def _has_gemini_key() -> bool:
    import config
    return bool(config.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY"))


def _has_gemma() -> bool:
    import config
    return bool(config.get("gemma_e4b") or config.get("gemma_31b"))


def estimate(viz: dict) -> dict:
    """견적서 필드를 예상해 dict로 반환.

    백엔드 선택은 메인 VLM과 똑같이 `config.vlm_backend`(+ Gemini 키 유무)를 따른다 —
    사용자가 UI에서 Gemini 키를 넣고 backend를 auto/gemini로 두면 견적 예상에도 Gemini API가
    쓰인다. 모델이 실패하거나 없으면 톤수·부피 기반 휴리스틱으로 보강한다. 절대 raise하지 않는다.
    """
    import config
    base = _heuristic(viz)                 # always a complete baseline
    backend = (config.get("vlm_backend") or "auto").lower()

    rec = None
    if backend == "heuristic":
        rec = None
    elif backend == "gemini" or (backend == "auto" and _has_gemini_key()):
        rec = _gemini(viz)                 # same key/setting as the main VLM stage
    elif backend in ("gemma_e4b", "gemma_31b", "gemma") or (backend == "auto" and _has_gemma()):
        rec = _gemma(viz)

    if rec:
        base.update({k: v for k, v in rec.items() if v not in (None, "")})
        log.info("견적 예상: %s", rec.get("source"))
    return base
