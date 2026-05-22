import os
import json
import requests
from typing import Optional

from rag.retriever import build_agent_rag_context

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL = "solar-pro3"

SYSTEM_PROMPT = """당신은 취업준비생의 주간 회고를 분석하는 감정·맥락 분석가입니다.

역할:
- KHT 회고 텍스트에서 감정과 맥락을 분류합니다.
- 번아웃 신호와 회복 신호를 판단합니다.
- 이번 주 핵심 고민 영역을 파악합니다.

응답 규칙:
- 반드시 아래 JSON 형식만 반환합니다. 설명 텍스트 없이 JSON만 출력하세요.
- intensity: 감정 강도 (low / medium / high)
- burnout_signal / recovery_signal: low / medium / high
- domain: 주요 고민 도메인 (자소서 / 면접 / 스펙 / 진로방향 / 대인관계 / 생활패턴 / 기타)

출력 JSON 형식:
{
  "emotion": {
    "primary": "주요 감정",
    "secondary": ["부감정1", "부감정2"],
    "intensity": "low | medium | high"
  },
  "context": {
    "domain": "주요 고민 도메인",
    "trigger": "감정 유발 요인 한 줄 요약"
  },
  "burnout_signal": "low | medium | high",
  "recovery_signal": "low | medium | high",
  "summary": "이번 주 상태 한 줄 요약",
  "graph_tags": ["Neo4j 태그용 감정/맥락 키워드 리스트"]
}"""


def run_analysis_agent(
    keep: list[str],
    hard: list[str],
    try_: list[str],
    api_key: Optional[str] = None,
    rag_context: Optional[str] = None,
    user_id: Optional[str] = None,
    week: Optional[int] = None,
) -> dict:
    """
    KHT 회고를 분석해 감정·맥락 분류 및 번아웃 신호를 반환합니다.

    Args:
        keep / hard / try_: KHT 항목 리스트
        api_key: Upstage API 키 (없으면 UPSTAGE_API_KEY 환경변수 사용)
        rag_context: Graph DB에서 검색한 유사 유저 패턴 (선택)
        user_id / week: Neo4j 저장용 메타데이터 (선택)
    """
    key = api_key or os.environ.get("SOLAR_API_KEY") or os.environ.get("UPSTAGE_API_KEY")
    if not key:
        raise ValueError("API 키가 필요합니다.")

    user_message = f"""이번 주 KHT 회고입니다.

[Keep - 잘 된 것]
{chr(10).join(f"- {item}" for item in keep) if keep else "- 없음"}

[Hard - 힘들었던 것]
{chr(10).join(f"- {item}" for item in hard) if hard else "- 없음"}

[Try - 다음 주 시도할 것]
{chr(10).join(f"- {item}" for item in try_) if try_ else "- 없음"}
"""

    if rag_context:
        user_message += f"\n[유사 유저 패턴 (Graph RAG 참고 컨텍스트)]\n{rag_context}\n"

    user_message += "\n감정과 맥락을 분석해주세요."

    response = requests.post(
        SOLAR_API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": SOLAR_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.3,
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
    sample_keep = ["알람 없이 기상", "산책 30분"]
    sample_hard = ["유튜브 보다 비교돼서 우울해짐", "자소서 한 줄도 못 씀"]
    sample_try = ["음악 틀어놓고 공부해보기"]

    api_key = input("Upstage API 키를 입력하세요: ").strip()
    print("\n분석 중...\n")
    result = run_analysis_agent(sample_keep, sample_hard, sample_try, api_key=api_key, user_id="test_user_01", week=1)
    print(json.dumps(result, ensure_ascii=False, indent=2))
