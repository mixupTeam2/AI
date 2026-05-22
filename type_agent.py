import os
import json
import requests
from typing import Optional

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL = "solar-pro3"

# 4축 정의
AXES = {
    "axis1": ("E", "B", "에너지 방향"),
    "axis2": ("C", "D", "관계 방향"),
    "axis3": ("S", "I", "사고 방향"),
    "axis4": ("G", "P", "동기 방향"),
}

CARE_TYPES = {
    "ECSG": ("설계하는 탐험가", "새로운 걸 체계적으로 만드는 사람"),
    "BDIP": ("깊이 파는 장인", "한 분야를 끝까지 갈아 임팩트를 내는 사람"),
    "ECIG": ("직관적 연결자", "사람과 아이디어를 감각으로 잇는 사람"),
    "BDSG": ("완성하는 전문가", "논리와 데이터로 결과물을 만드는 사람"),
    "ECSP": ("기획하는 리더", "체계적으로 팀을 이끌어 임팩트를 내는 사람"),
    # 나머지 11개 조합 — 미정의 유형은 아래 fallback으로 처리
}

DESCRIBE_PROMPT = """당신은 CareType 취업준비생 유형 분석가입니다.
누적 점수 기반으로 결정된 CareType 코드와 4축 성향을 바탕으로 유형 설명을 작성해주세요.

반드시 아래 JSON만 반환하세요:
{
  "type_name": "유형명 (3~5자)",
  "description": "이 유형 한 줄 설명",
  "strengths": ["강점1", "강점2", "강점3"],
  "care_tip": "이 유형에게 맞는 취준 케어 팁 한 줄"
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


def _calculate_code(cumulative_scores: dict) -> str:
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
                           [{"axis1": {"score": 1, ...}, "axis2": {...}, ...}, ...]
        api_key: Upstage API 키 (없으면 UPSTAGE_API_KEY 환경변수 사용)
        user_id: Neo4j 저장용 메타데이터 (선택)

    Returns:
        {
          "code": "ECSG",
          "weeks_accumulated": 10,
          "axis_totals": {"axis1": 5, "axis2": -2, "axis3": 3, "axis4": 1},
          "axis_labels": {"axis1": "E", "axis2": "D", "axis3": "S", "axis4": "G"},
          "type_name": "설계하는 탐험가",
          "description": "...",
          "strengths": [...],
          "care_tip": "...",
          "is_complete": True  # 10주 이상이면 True
        }
    """
    key = api_key or os.environ.get("UPSTAGE_API_KEY")
    if not key:
        raise ValueError("API 키가 필요합니다.")

    weeks = len(cumulative_scores)
    is_complete = weeks >= 10

    # 축별 누적 합산
    axis_totals = {}
    axis_labels = {}
    for axis_key, (pos_label, neg_label, axis_name) in AXES.items():
        total = sum(week.get(axis_key, {}).get("score", 0) for week in cumulative_scores)
        axis_totals[axis_key] = total
        axis_labels[axis_key] = pos_label if total >= 0 else neg_label

    code = "".join(axis_labels[k] for k in ["axis1", "axis2", "axis3", "axis4"])

    # 미리 정의된 유형이면 바로 사용, 없으면 Solar로 생성
    if code in CARE_TYPES:
        type_name, description = CARE_TYPES[code]
        type_info = {"type_name": type_name, "description": description, "strengths": [], "care_tip": ""}
    else:
        axis_summary = "\n".join(
            f"- {name}({pos}/{neg}): 누적점수 {axis_totals[k]} → {axis_labels[k]}형"
            for k, (pos, neg, name) in AXES.items()
        )
        raw = _call_solar([
            {"role": "system", "content": DESCRIBE_PROMPT},
            {"role": "user", "content": f"CareType 코드: {code}\n\n4축 성향:\n{axis_summary}"},
        ], key)
        type_info = _parse_json(raw)

    return {
        "code": code,
        "weeks_accumulated": weeks,
        "is_complete": is_complete,
        "axis_totals": axis_totals,
        "axis_labels": axis_labels,
        **type_info,
        "meta": {"user_id": user_id},
    }


if __name__ == "__main__":
    # 10주치 샘플 데이터
    sample_scores = [
        {"axis1": {"score": 2}, "axis2": {"score": 1}, "axis3": {"score": 1}, "axis4": {"score": 2}},
        {"axis1": {"score": 1}, "axis2": {"score": 0}, "axis3": {"score": 2}, "axis4": {"score": 1}},
        {"axis1": {"score": 2}, "axis2": {"score": 1}, "axis3": {"score": 1}, "axis4": {"score": 0}},
        {"axis1": {"score": 0}, "axis2": {"score": -1}, "axis3": {"score": 2}, "axis4": {"score": 1}},
        {"axis1": {"score": 1}, "axis2": {"score": 1}, "axis3": {"score": 0}, "axis4": {"score": 2}},
        {"axis1": {"score": 2}, "axis2": {"score": 0}, "axis3": {"score": 1}, "axis4": {"score": 1}},
        {"axis1": {"score": 1}, "axis2": {"score": 1}, "axis3": {"score": 2}, "axis4": {"score": 0}},
        {"axis1": {"score": -1}, "axis2": {"score": 2}, "axis3": {"score": 1}, "axis4": {"score": 1}},
        {"axis1": {"score": 2}, "axis2": {"score": 1}, "axis3": {"score": 0}, "axis4": {"score": 2}},
        {"axis1": {"score": 1}, "axis2": {"score": 0}, "axis3": {"score": 1}, "axis4": {"score": 1}},
    ]

    api_key = input("Upstage API 키를 입력하세요: ").strip()
    print("\nCareType 생성 중...\n")
    result = run_type_agent(sample_scores, api_key=api_key, user_id="test_user_01")
    print(json.dumps(result, ensure_ascii=False, indent=2))
