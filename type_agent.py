import os
import json
import requests
from typing import Optional

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL = "solar-pro3"

# 4축 정의 (양수 = 앞 유형, 음수 = 뒤 유형)
AXES = {
    "axis1": ("E", "A", "감정 대응 방식"),   # E(외향/+) vs A(내향/-)
    "axis2": ("P", "X", "행동 전략"),         # P(계획/+) vs X(탐색/-)
    "axis3": ("G", "R", "동기 원천"),         # G(성장/+) vs R(인정/-)
    "axis4": ("C", "S", "스트레스 반응"),     # C(지속/+) vs S(전환/-)
}

CARE_TYPES = {
    "APGC": ("묵묵한 설계자",     "혼자 계획을 세우고 묵묵히 성장해나가는 유형"),
    "APGS": ("유연한 전략가",     "내면 중심이지만 막히면 유연하게 방향을 바꾸는 유형"),
    "APRC": ("인정받는 완벽주의자","혼자 철저히 준비해 결과로 인정받으려는 유형"),
    "APRS": ("현실적인 조율자",   "평가를 의식하면서도 현실적으로 방향을 조율하는 유형"),
    "AXGC": ("고독한 탐험가",     "혼자 다양한 시도를 하며 배움에서 동력을 얻는 유형"),
    "AXGS": ("자유로운 실험가",   "혼자 이것저것 시도하다 막히면 쉽게 전환하는 유형"),
    "AXRC": ("승부사형 도전자",   "혼자 도전하며 결과와 인정을 위해 버티는 유형"),
    "AXRS": ("감각적인 방랑자",   "혼자 경험을 쌓으며 반응에 따라 방향을 바꾸는 유형"),
    "EPGC": ("열정적인 추진자",   "공유하며 동기를 얻고 성장을 위해 끝까지 밀어붙이는 유형"),
    "EPGS": ("공감형 리더",       "함께 계획하고 성장하되 상황에 따라 유연하게 전환하는 유형"),
    "EPRC": ("무대형 실행가",     "표출하며 에너지를 얻고 인정받기 위해 계획을 고수하는 유형"),
    "EPRS": ("유쾌한 전환자",     "공유하며 인정받되 막히면 빠르게 방향을 전환하는 유형"),
    "EXGC": ("에너지형 탐색자",   "다양한 시도를 공유하며 배움 자체에서 동력을 얻는 유형"),
    "EXGS": ("네트워크형 모험가", "사람들과 함께 탐색하며 막히면 자연스럽게 전환하는 유형"),
    "EXRC": ("존재감형 도전자",   "표출하며 다양하게 도전해 인정받고자 끝까지 버티는 유형"),
    "EXRS": ("감성적인 흐름형",   "감각과 공유로 이것저것 시도하다 흐름에 따라 전환하는 유형"),
}

DESCRIBE_PROMPT = """당신은 CareType 취업준비생 유형 분석가입니다.
누적 점수 기반으로 결정된 CareType 코드와 4축 성향을 바탕으로 케어 팁을 작성해주세요.

반드시 아래 JSON만 반환하세요:
{
  "strengths": ["강점1", "강점2", "강점3"],
  "care_tip": "이 유형에게 맞는 취준 케어 팁 한 줄",
  "risk": "이 유형이 주의해야 할 번아웃 패턴 한 줄"
}"""


def _call_solar(messages: list[dict], api_key: str) -> str:
    response = requests.post(
        SOLAR_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": SOLAR_MODEL, "messages": messages, "temperature": 0.5},
        timeout=30,
    )
    if not response.ok:
        print("에러 응답:", response.status_code, response.text)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _parse_json(raw: str) -> dict:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def _calculate_code(cumulative_scores: list[dict]) -> str:
    """누적 점수 합산 → 4자리 CareType 코드 결정"""
    code = ""
    for axis_key, (pos_label, neg_label, _) in AXES.items():
        total = sum(week.get(axis_key, {}).get("score", 0) for week in cumulative_scores)
        code += pos_label if total >= 0 else neg_label
    return code


def run_type_agent(
    cumulative_scores: list[dict],
    api_key: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """
    누적 주간 점수로 CareType 코드를 생성합니다.

    Args:
        cumulative_scores: 주차별 pattern_agent 결과 리스트
                           [{"axis1": {"score": 1}, "axis2": {"score": -1}, ...}, ...]
        api_key: Upstage API 키 (없으면 UPSTAGE_API_KEY 환경변수 사용)
        user_id: Neo4j 저장용 메타데이터 (선택)

    Returns:
        {
          "code": "APGC",
          "type_name": "묵묵한 설계자",
          "description": "...",
          "strengths": [...],
          "care_tip": "...",
          "risk": "...",
          "weeks_accumulated": 10,
          "is_complete": True,
          "axis_totals": {"axis1": 5, ...},
          "axis_labels": {"axis1": "E", ...}
        }
    """
    key = api_key or os.environ.get("SOLAR_API_KEY") or os.environ.get("UPSTAGE_API_KEY")
    if not key:
        raise ValueError("API 키가 필요합니다.")

    weeks = len(cumulative_scores)
    is_complete = weeks >= 10

    # 축별 누적 합산
    axis_totals = {}
    axis_labels = {}
    for axis_key, (pos_label, neg_label, _) in AXES.items():
        total = sum(week.get(axis_key, {}).get("score", 0) for week in cumulative_scores)
        axis_totals[axis_key] = total
        axis_labels[axis_key] = pos_label if total >= 0 else neg_label

    code = "".join(axis_labels[k] for k in ["axis1", "axis2", "axis3", "axis4"])
    type_name, description = CARE_TYPES.get(code, ("알 수 없는 유형", ""))

    # Solar로 강점·케어팁·리스크 생성
    axis_summary = "\n".join(
        f"- {name}({pos}/{neg}): 누적점수 {axis_totals[k]} → {axis_labels[k]}형"
        for k, (pos, neg, name) in AXES.items()
    )
    extra = _parse_json(_call_solar([
        {"role": "system", "content": DESCRIBE_PROMPT},
        {"role": "user", "content": f"CareType 코드: {code} ({type_name})\n\n4축 성향:\n{axis_summary}"},
    ], key))

    return {
        "code": code,
        "type_name": type_name,
        "description": description,
        "strengths": extra.get("strengths", []),
        "care_tip": extra.get("care_tip", ""),
        "risk": extra.get("risk", ""),
        "weeks_accumulated": weeks,
        "is_complete": is_complete,
        "axis_totals": axis_totals,
        "axis_labels": axis_labels,
        "meta": {"user_id": user_id},
    }


if __name__ == "__main__":
    sample_scores = [
        {"axis1": {"score": -1}, "axis2": {"score": 2}, "axis3": {"score": 1}, "axis4": {"score": 2}},
        {"axis1": {"score": -2}, "axis2": {"score": 1}, "axis3": {"score": 2}, "axis4": {"score": 1}},
        {"axis1": {"score": 0},  "axis2": {"score": 2}, "axis3": {"score": 1}, "axis4": {"score": 2}},
        {"axis1": {"score": -1}, "axis2": {"score": 1}, "axis3": {"score": 2}, "axis4": {"score": 1}},
        {"axis1": {"score": -2}, "axis2": {"score": 2}, "axis3": {"score": 1}, "axis4": {"score": 2}},
        {"axis1": {"score": -1}, "axis2": {"score": 1}, "axis3": {"score": 2}, "axis4": {"score": 1}},
        {"axis1": {"score": 0},  "axis2": {"score": 2}, "axis3": {"score": 1}, "axis4": {"score": 2}},
        {"axis1": {"score": -1}, "axis2": {"score": 1}, "axis3": {"score": 1}, "axis4": {"score": 1}},
        {"axis1": {"score": -2}, "axis2": {"score": 2}, "axis3": {"score": 2}, "axis4": {"score": 2}},
        {"axis1": {"score": -1}, "axis2": {"score": 1}, "axis3": {"score": 1}, "axis4": {"score": 1}},
    ]  # 예상 결과: APGC (묵묵한 설계자)

    api_key = input("Upstage API 키를 입력하세요: ").strip()
    print("\nCareType 생성 중...\n")
    result = run_type_agent(sample_scores, api_key=api_key, user_id="test_user_01")
    print(json.dumps(result, ensure_ascii=False, indent=2))
