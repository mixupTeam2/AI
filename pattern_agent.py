import os
import json
import requests
from typing import Optional

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL = "solar-pro3"

SYSTEM_PROMPT = """당신은 취업준비생의 KHT 회고에서 CareType 4개 축 점수를 채점하는 분석가입니다.

## 4개 축 기준

### 축1 — 에너지 방향: E(탐험형) vs B(구축형)
- E 신호어: "새로운 걸 시도", "이것저것 관심", "기획이 재밌다", "도전", "다양한"
- B 신호어: "끝까지 완성", "결과물", "실행이 먼저", "마무리", "완성도"

### 축2 — 관계 방향: C(연결형) vs D(심화형)
- C 신호어: "팀워크", "사람들과", "설득", "네트워킹", "함께"
- D 신호어: "혼자 집중", "전문성", "깊이", "한 분야", "연구"

### 축3 — 사고 방향: S(구조형) vs I(직관형)
- S 신호어: "정리", "계획", "근거", "체계", "데이터", "논리"
- I 신호어: "느낌", "흐름", "맥락", "감각", "직관"

### 축4 — 동기 방향: G(성장형) vs P(영향형)
- G 신호어: "배우는 것", "성장", "발전", "실력", "내가 나아지는"
- P 신호어: "도움", "임팩트", "세상을 바꾸", "누군가에게", "영향"

## 채점 규칙
- 각 축별로 -2 ~ +2 사이 정수 점수를 부여합니다.
- 양수 = 앞 유형 (E/C/S/G), 음수 = 뒤 유형 (B/D/I/P)
- 신호어가 없으면 0점
- 반드시 아래 JSON만 반환하세요. 설명 없이 JSON만 출력하세요.

출력 JSON 형식:
{
  "axis1": {"score": 정수(-2~+2), "detected": ["감지된 신호어"], "reason": "판단 근거 한 줄"},
  "axis2": {"score": 정수(-2~+2), "detected": ["감지된 신호어"], "reason": "판단 근거 한 줄"},
  "axis3": {"score": 정수(-2~+2), "detected": ["감지된 신호어"], "reason": "판단 근거 한 줄"},
  "axis4": {"score": 정수(-2~+2), "detected": ["감지된 신호어"], "reason": "판단 근거 한 줄"}
}"""


def run_pattern_agent(
    keep: list[str],
    hard: list[str],
    try_: list[str],
    api_key: Optional[str] = None,
    rag_context: Optional[str] = None,
    user_id: Optional[str] = None,
    week: Optional[int] = None,
) -> dict:
    """
    KHT 회고에서 CareType 4축 점수를 채점합니다.

    Returns:
        {
          "axis1": {"score": 1, "detected": [...], "reason": "..."},  # E(+) vs B(-)
          "axis2": {"score": -1, "detected": [...], "reason": "..."},  # C(+) vs D(-)
          "axis3": {"score": 2, "detected": [...], "reason": "..."},  # S(+) vs I(-)
          "axis4": {"score": 0, "detected": [...], "reason": "..."},  # G(+) vs P(-)
          "meta": {"user_id": ..., "week": ...}
        }
    """
    key = api_key or os.environ.get("SOLAR_API_KEY") or os.environ.get("UPSTAGE_API_KEY")
    if not key:
        raise ValueError("API 키가 필요합니다.")

    user_message = f"""이번 주 KHT 회고입니다. 4개 축 점수를 채점해주세요.

[Keep - 잘 된 것]
{chr(10).join(f"- {item}" for item in keep) if keep else "- 없음"}

[Hard - 힘들었던 것]
{chr(10).join(f"- {item}" for item in hard) if hard else "- 없음"}

[Try - 다음 주 시도할 것]
{chr(10).join(f"- {item}" for item in try_) if try_ else "- 없음"}
"""
    if rag_context:
        user_message += f"\n[추가 컨텍스트]\n{rag_context}\n"

    response = requests.post(
        SOLAR_API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": SOLAR_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.2,
        },
        timeout=30,
    )
    if not response.ok:
        print("에러 응답:", response.status_code, response.text)
    response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    result = json.loads(raw)
    result["meta"] = {"user_id": user_id, "week": week}
    return result


if __name__ == "__main__":
    sample_keep = ["새로운 기획 아이디어 떠올림", "팀원이랑 얘기하니 좋았음"]
    sample_hard = ["계획을 못 지켰음", "결과물이 없는 느낌"]
    sample_try = ["데이터 분석 공부해보기"]

    api_key = input("Upstage API 키를 입력하세요: ").strip()
    print("\n채점 중...\n")
    result = run_pattern_agent(sample_keep, sample_hard, sample_try, api_key=api_key, user_id="test_user_01", week=1)
    print(json.dumps(result, ensure_ascii=False, indent=2))
