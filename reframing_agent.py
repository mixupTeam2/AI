import os
import json
import requests
from typing import Optional

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL = "solar-pro3"

SYSTEM_PROMPT = """당신은 취업준비생의 주간 회고를 분석하는 리프레이밍 코치입니다.

역할:
- Hard(힘들었던 것) 항목을 강점 언어로 전환해 리프레이밍 피드백을 제공합니다.
- 각 Hard 항목에서 강점 키워드(2~3개)를 추출합니다.
- 번아웃 신호 수준을 판단합니다 (low / medium / high).
- 유사 유저의 패턴이 제공된 경우 참고해 더 맥락적인 피드백을 작성합니다.

응답 규칙:
- 반드시 아래 JSON 형식만 반환합니다. 설명 텍스트 없이 JSON만 출력하세요.
- reframed 메시지는 따뜻하고 구체적으로, 1~2문장으로 작성합니다.
- strength_keywords는 한국어로 2~3개 작성합니다.
- emotion_tags는 Hard 항목에서 감지된 감정을 1~3개 추출합니다.
- burnout_signal: Hard 항목의 감정 강도와 개수를 기준으로 판단합니다.

출력 JSON 형식:
{
  "reframing": [
    {
      "original": "원문 Hard 항목",
      "reframed": "리프레이밍 메시지",
      "strength_keywords": ["키워드1", "키워드2"],
      "emotion_tags": ["감정1", "감정2"]
    }
  ],
  "overall_message": "이번 주 전체 격려 메시지 (2~3문장)",
  "burnout_signal": "low | medium | high",
  "graph_keywords": ["전체 강점 키워드 통합 리스트 (Neo4j 노드용)"]
}"""


def run_reframing_agent(
    keep: list[str],
    hard: list[str],
    try_: list[str],
    api_key: Optional[str] = None,
    rag_context: Optional[str] = None,
    user_id: Optional[str] = None,
    week: Optional[int] = None,
) -> dict:
    """
    KHT 회고 입력을 받아 리프레이밍 피드백 + 강점 키워드를 JSON으로 반환합니다.

    Args:
        keep: Keep 항목 리스트
        hard: Hard 항목 리스트 (핵심 분석 대상)
        try_: Try 항목 리스트
        api_key: Upstage API 키 (없으면 환경변수 UPSTAGE_API_KEY 사용)
        rag_context: Graph DB에서 검색한 유사 유저 패턴 텍스트 (선택)
                     예: "유사 유저들은 비교 트리거를 SNS 차단으로 해결했습니다."
        user_id: 유저 식별자 — 결과에 메타데이터로 포함됨 (Neo4j 저장용, 선택)
        week: 회고 회차 번호 (선택)

    Returns:
        {
          "reframing": [{"original", "reframed", "strength_keywords", "emotion_tags"}, ...],
          "overall_message": str,
          "burnout_signal": "low" | "medium" | "high",
          "graph_keywords": [...],   # Neo4j 키워드 노드 생성용
          "meta": {"user_id", "week"}  # Graph DB 저장 시 사용
        }
    """
    key = api_key or os.environ.get("SOLAR_API_KEY") or os.environ.get("UPSTAGE_API_KEY")
    if not key:
        raise ValueError("API 키가 필요합니다. UPSTAGE_API_KEY 환경변수를 설정하거나 api_key 인자를 전달하세요.")

    user_message = f"""이번 주 KHT 회고입니다.

[Keep - 잘 된 것]
{chr(10).join(f"- {item}" for item in keep) if keep else "- 없음"}

[Hard - 힘들었던 것]
{chr(10).join(f"- {item}" for item in hard) if hard else "- 없음"}

[Try - 다음 주 시도할 것]
{chr(10).join(f"- {item}" for item in try_) if try_ else "- 없음"}
"""

    # Graph RAG 컨텍스트가 있으면 프롬프트에 주입
    if rag_context:
        user_message += f"""
[유사 유저 패턴 (Graph RAG 참고 컨텍스트)]
{rag_context}
"""

    user_message += "\nHard 항목을 리프레이밍해주세요."

    response = requests.post(
        SOLAR_API_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": SOLAR_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.7,
        },
        timeout=30,
    )
    if not response.ok:
        print("에러 응답:", response.status_code, response.text)
    response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"].strip()

    # 마크다운 코드블록 제거 (```json ... ```)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    result = json.loads(raw)

    # Neo4j 저장 시 필요한 메타데이터 추가
    result["meta"] = {
        "user_id": user_id,
        "week": week,
    }

    return result


if __name__ == "__main__":
    sample_keep = ["알람 없이 기상", "산책 30분"]
    sample_hard = ["유튜브 보다 비교돼서 우울해짐", "자소서 한 줄도 못 씀"]
    sample_try = ["음악 틀어놓고 공부해보기"]

    # Graph RAG 연결 전 테스트용 더미 컨텍스트
    sample_rag_context = None
    # 나중에 Neo4j에서 가져오면 이렇게 씁니다:
    # sample_rag_context = "비슷한 상황의 유저들은 3주차에 비교 감정이 줄었고, SNS 사용 시간 제한이 효과적이었습니다."

    api_key = input("Upstage API 키를 입력하세요: ").strip()

    print("\n분석 중...\n")
    result = run_reframing_agent(
        sample_keep,
        sample_hard,
        sample_try,
        api_key=api_key,
        rag_context=sample_rag_context,
        user_id="test_user_01",
        week=1,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
