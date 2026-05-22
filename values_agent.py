import os
import json
import requests
from typing import Optional

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL = "solar-pro3"

SYSTEM_PROMPT = """당신은 취업준비생의 가치관 변화를 추적하는 전문가입니다.

역할:
- KHT 회고 텍스트에서 가치관 관련 신호를 감지합니다.
- 현재 가치관 우선순위와 비교해 변화 여부를 판단합니다.
- 변화가 감지되면 업데이트 제안 메시지를 생성합니다.

가치관 항목: 근무지 / 워라밸 / 도메인 / 직무 / 연봉 / 기업규모 / 안정성

응답 규칙:
- 반드시 아래 JSON 형식만 반환합니다. 설명 텍스트 없이 JSON만 출력하세요.
- change_detected: 가치관 변화 신호가 있으면 true
- suggested_priority: 변화가 감지된 항목만 포함 (변화 없으면 빈 객체)
- update_message: 변화가 없으면 null

출력 JSON 형식:
{
  "detected_signals": ["회고 텍스트에서 감지된 가치관 신호 문장들"],
  "change_detected": true,
  "changed_values": ["변화가 감지된 가치관 항목"],
  "suggested_priority": {
    "근무지": 1,
    "워라밸": 2
  },
  "update_message": "가치관이 바뀐 것 같아요. 업데이트할까요? (없으면 null)",
  "graph_tags": ["Neo4j 태그용 가치관 키워드 리스트"]
}"""


def run_values_agent(
    keep: list[str],
    hard: list[str],
    try_: list[str],
    current_priority: dict[str, int],
    api_key: Optional[str] = None,
    rag_context: Optional[str] = None,
    user_id: Optional[str] = None,
    week: Optional[int] = None,
) -> dict:
    """
    KHT 회고에서 가치관 변화를 감지하고 우선순위 업데이트를 제안합니다.

    Args:
        keep / hard / try_: KHT 항목 리스트
        current_priority: 현재 가치관 우선순위 {"근무지": 1, "워라밸": 2, ...}
        api_key: Upstage API 키 (없으면 UPSTAGE_API_KEY 환경변수 사용)
        rag_context: Graph DB에서 검색한 유사 유저 패턴 (선택)
        user_id / week: Neo4j 저장용 메타데이터 (선택)
    """
    key = api_key or os.environ.get("SOLAR_API_KEY") or os.environ.get("UPSTAGE_API_KEY")
    if not key:
        raise ValueError("API 키가 필요합니다.")

    priority_text = "\n".join(
        f"{rank}순위: {value}"
        for value, rank in sorted(current_priority.items(), key=lambda x: x[1])
    )

    user_message = f"""이번 주 KHT 회고입니다.

[Keep - 잘 된 것]
{chr(10).join(f"- {item}" for item in keep) if keep else "- 없음"}

[Hard - 힘들었던 것]
{chr(10).join(f"- {item}" for item in hard) if hard else "- 없음"}

[Try - 다음 주 시도할 것]
{chr(10).join(f"- {item}" for item in try_) if try_ else "- 없음"}

[현재 가치관 우선순위]
{priority_text}
"""

    if rag_context:
        user_message += f"\n[유사 유저 패턴 (Graph RAG 참고 컨텍스트)]\n{rag_context}\n"

    user_message += "\n회고 텍스트에서 가치관 변화 신호를 감지해주세요."

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
    sample_hard = ["연봉보다 집 근처가 더 중요한 것 같다는 생각이 자꾸 듦", "대기업 목표인데 맞는건지 모르겠음"]
    sample_try = ["중소기업도 검색해보기"]

    sample_priority = {
        "근무지": 6,
        "워라밸": 2,
        "도메인": 3,
        "직무": 4,
        "연봉": 1,
        "기업규모": 5,
        "안정성": 7,
    }

    api_key = input("Upstage API 키를 입력하세요: ").strip()
    print("\n분석 중...\n")
    result = run_values_agent(
        sample_keep, sample_hard, sample_try,
        current_priority=sample_priority,
        api_key=api_key,
        user_id="test_user_01",
        week=3,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
