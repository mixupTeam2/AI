import os
import json
import requests
from typing import Optional

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL = "solar-pro3"

SYSTEM_PROMPT = """당신은 취업준비생의 KHT 회고에서 CareType 4개 축 점수를 채점하는 분석가입니다.

## 4개 축 기준

### 축1 — 감정 대응 방식: E(외향) vs A(내향)
- E(+) 신호어: "표출", "공유", "얘기하니", "사람들과 이야기", "털어놓으니", "말하고 나서"
- A(-) 신호어: "혼자 소화", "혼자서", "내면", "조용히", "혼자 정리", "혼자 해결"

### 축2 — 행동 전략: P(계획) vs X(탐색)
- P(+) 신호어: "루틴", "계획", "목표", "정해진", "체계적", "순서대로", "일정대로"
- X(-) 신호어: "새로운", "시도", "이것저것", "경험", "해봤더니", "도전해봤는데"

### 축3 — 동기 원천: G(성장) vs R(인정)
- G(+) 신호어: "배우는 것", "성장", "발전", "실력", "배움 자체", "내가 나아지는"
- R(-) 신호어: "결과", "평가", "인정", "타인 반응", "피드백", "합격", "점수"

### 축4 — 스트레스 반응: C(지속) vs S(전환)
- C(+) 신호어: "버텼", "유지", "꾸준히", "계속", "포기 안", "끝까지"
- S(-) 신호어: "방향 바꿨", "쉬었", "전환", "잠깐 멈추고", "다른 방법", "바꿔보니"

## 채점 규칙
- 각 축별로 -2 ~ +2 사이 정수 점수를 부여합니다.
- 양수 = 앞 유형 (E/P/G/C), 음수 = 뒤 유형 (A/X/R/S)
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
    key = api_key or os.environ.get("UPSTAGE_API_KEY")
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
