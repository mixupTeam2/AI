import os
import json
import requests
from typing import Optional

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL = "solar-pro3"

SYSTEM_PROMPT = """당신은 취업준비생의 KHT 회고에서 CareType 4개 축 점수를 채점하는 분석가입니다.

## 4개 축 기준 (의미와 맥락으로 판단하세요 — 정확한 단어가 없어도 됩니다)

### 축1 — 감정 대응 방식: E(외향/+) vs A(내향/-)
- E: 감정을 밖으로 표출하거나 공유해서 해소하는 유형
  예) 누군가에게 털어놓음, 사람들과 이야기함, 감정을 표현함
- A: 혼자 감정을 소화하고 내면에서 정리하는 유형
  예) 혼자 산책, 혼자 정리, 내면에서 해결, 조용히 시간을 보냄

### 축2 — 행동 전략: P(계획/+) vs X(탐색/-)
- P: 루틴·목표·순서 중심으로 행동하는 유형
  예) 계획 세움, 루틴 유지, 목표 기반 실행, 정해진 시간에 공부
- X: 새로운 시도·경험·실험을 통해 행동하는 유형
  예) 새로운 방법 시도, 이것저것 해봄, 도전적 접근, 환경 바꿔보기

### 축3 — 동기 원천: G(성장/+) vs R(인정/-)
- G: 배움과 발전 자체에서 동기를 얻는 유형
  예) 배우는 것 자체가 즐거움, 실력 향상에 집중, 성장 중시
- R: 결과·평가·타인 반응에서 동기를 얻는 유형
  예) 합격이 목표, 남들 반응 신경 씀, 비교하며 동기 부여, 인정받고 싶음

### 축4 — 스트레스 반응: C(지속/+) vs S(전환/-)
- C: 힘들어도 버티며 유지하는 유형
  예) 힘들지만 계속함, 포기하지 않음, 꾸준히 유지
- S: 막히면 방향을 바꾸거나 쉬어가는 유형
  예) 다른 방법 시도, 잠깐 쉬고 재도전, 유연하게 전환

## 채점 규칙
- 각 축별로 -2 ~ +2 사이 정수 점수를 부여합니다.
- 양수 = 앞 유형 (E/P/G/C), 음수 = 뒤 유형 (A/X/R/S)
- 명확한 신호가 없으면 약한 점수(±1) 또는 0점을 부여합니다.
- 신호어가 정확히 없어도 맥락과 의미로 판단하세요.
- 반드시 아래 JSON만 반환하세요. 설명 없이 JSON만 출력하세요.

출력 JSON 형식:
{
  "axis1": {"score": 정수(-2~+2), "detected": ["판단 근거가 된 표현"], "reason": "판단 근거 한 줄"},
  "axis2": {"score": 정수(-2~+2), "detected": ["판단 근거가 된 표현"], "reason": "판단 근거 한 줄"},
  "axis3": {"score": 정수(-2~+2), "detected": ["판단 근거가 된 표현"], "reason": "판단 근거 한 줄"},
  "axis4": {"score": 정수(-2~+2), "detected": ["판단 근거가 된 표현"], "reason": "판단 근거 한 줄"}
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
          "axis1": {"score": 1, "detected": [...], "reason": "..."},   # E(+) vs A(-)
          "axis2": {"score": -1, "detected": [...], "reason": "..."},  # P(+) vs X(-)
          "axis3": {"score": 2, "detected": [...], "reason": "..."},   # G(+) vs R(-)
          "axis4": {"score": 0, "detected": [...], "reason": "..."},   # C(+) vs S(-)
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
