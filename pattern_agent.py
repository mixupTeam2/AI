import os
import json
import requests
from typing import Optional

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL = "solar-pro3"

SYSTEM_PROMPT = """당신은 취업준비생의 회고 데이터를 시계열로 분석하는 패턴 추적 전문가입니다.

역할:
- 누적된 주간 회고 데이터에서 감정·에너지 변화 트렌드를 감지합니다.
- 반복되는 고민 패턴과 회복 신호를 파악합니다.
- 현재 타이밍에 맞는 유동적 솔루션을 제안합니다.

응답 규칙:
- 반드시 아래 JSON 형식만 반환합니다. 설명 텍스트 없이 JSON만 출력하세요.
- trend: improving / declining / stable
- 누적 데이터가 1회뿐이면 trend는 "stable", recurring_themes는 빈 리스트로 반환합니다.

출력 JSON 형식:
{
  "trend": "improving | declining | stable",
  "energy_level": "low | medium | high",
  "recurring_themes": ["반복되는 고민 테마1", "테마2"],
  "turning_point": "변화 시점 또는 null",
  "insight": "패턴 기반 인사이트 (2문장)",
  "action_suggestion": "현재 타이밍에 맞는 유동적 솔루션 (1문장)",
  "graph_tags": ["Neo4j 태그용 패턴 키워드 리스트"]
}"""


def run_pattern_agent(
    current_week: dict,
    history: list[dict],
    api_key: Optional[str] = None,
    rag_context: Optional[str] = None,
    user_id: Optional[str] = None,
    week: Optional[int] = None,
) -> dict:
    """
    현재 회고 + 누적 히스토리를 분석해 패턴과 유동적 솔루션을 반환합니다.

    Args:
        current_week: 이번 주 KHT {"keep": [...], "hard": [...], "try": [...]}
        history: 이전 주 회고 리스트 [{"week": 1, "keep": [...], "hard": [...], "try": [...]}, ...]
                 빈 리스트여도 동작합니다 (1회차인 경우).
        api_key: Upstage API 키 (없으면 UPSTAGE_API_KEY 환경변수 사용)
        rag_context: Graph DB에서 검색한 유사 유저 패턴 (선택)
        user_id / week: Neo4j 저장용 메타데이터 (선택)
    """
    key = api_key or os.environ.get("UPSTAGE_API_KEY")
    if not key:
        raise ValueError("API 키가 필요합니다.")

    history_text = ""
    if history:
        for entry in history:
            w = entry.get("week", "?")
            history_text += f"\n[{w}주차]\n"
            history_text += f"Keep: {', '.join(entry.get('keep', []))}\n"
            history_text += f"Hard: {', '.join(entry.get('hard', []))}\n"
            history_text += f"Try: {', '.join(entry.get('try', []))}\n"
    else:
        history_text = "없음 (첫 번째 회고)"

    user_message = f"""누적 회고 데이터입니다.

[이전 회고 히스토리]
{history_text}

[이번 주 회고]
Keep: {', '.join(current_week.get('keep', []))}
Hard: {', '.join(current_week.get('hard', []))}
Try: {', '.join(current_week.get('try', []))}
"""

    if rag_context:
        user_message += f"\n[유사 유저 패턴 (Graph RAG 참고 컨텍스트)]\n{rag_context}\n"

    user_message += "\n패턴을 분석하고 유동적 솔루션을 제안해주세요."

    response = requests.post(
        SOLAR_API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": SOLAR_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.5,
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
    current = {
        "keep": ["알람 없이 기상", "산책 30분"],
        "hard": ["유튜브 보다 비교돼서 우울해짐", "자소서 한 줄도 못 씀"],
        "try": ["음악 틀어놓고 공부해보기"],
    }
    history = [
        {"week": 1, "keep": ["운동"], "hard": ["면접 떨어짐", "의욕 없음"], "try": ["독서"]},
        {"week": 2, "keep": ["독서 완료"], "hard": ["비교감", "집중 안 됨"], "try": ["카페 공부"]},
    ]

    api_key = input("Upstage API 키를 입력하세요: ").strip()
    print("\n분석 중...\n")
    result = run_pattern_agent(current, history, api_key=api_key, user_id="test_user_01", week=3)
    print(json.dumps(result, ensure_ascii=False, indent=2))
