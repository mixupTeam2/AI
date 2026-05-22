import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from analysis_agent import run_analysis_agent
from reframing_agent import run_reframing_agent
from pattern_agent import run_pattern_agent
from values_agent import run_values_agent

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL = "solar-pro3"

SUPERVISOR_SYSTEM_PROMPT = """당신은 취업준비생 케어 멀티에이전트 시스템의 슈퍼바이저입니다.

역할:
- 4개 전문 에이전트(분석/리프레이밍/패턴추적/가치관추적)의 결과를 종합합니다.
- 이번 주 유저 상태를 통합적으로 판단하고 핵심 메시지를 도출합니다.
- 가장 시급한 케어 포인트 1가지를 결정합니다.

응답 규칙:
- 반드시 아래 JSON 형식만 반환합니다. 설명 텍스트 없이 JSON만 출력하세요.

출력 JSON 형식:
{
  "care_point": "이번 주 가장 시급한 케어 포인트 한 줄",
  "main_message": "유저에게 전달할 핵심 메시지 (2~3문장, 따뜻하고 구체적으로)",
  "action": "지금 당장 시도할 수 있는 행동 1가지",
  "next_week_focus": "다음 주 집중할 방향 한 줄",
  "agent_weights": {
    "analysis": "이번 주 분석 에이전트 기여도 요약 한 줄",
    "reframing": "이번 주 리프레이밍 에이전트 기여도 요약 한 줄",
    "pattern": "이번 주 패턴 에이전트 기여도 요약 한 줄",
    "values": "이번 주 가치관 에이전트 기여도 요약 한 줄"
  }
}"""


def _call_solar(messages: list[dict], api_key: str, temperature: float = 0.5) -> str:
    response = requests.post(
        SOLAR_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": SOLAR_MODEL, "messages": messages, "temperature": temperature},
        timeout=30,
    )
    if not response.ok:
        print("슈퍼바이저 에러:", response.status_code, response.text)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _parse_json(raw: str) -> dict:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def _supervisor_synthesize(
    analysis: dict,
    reframing: dict,
    pattern: dict,
    values: dict,
    api_key: str,
) -> dict:
    """슈퍼바이저가 4개 에이전트 결과를 Solar로 종합합니다."""
    agent_results = json.dumps(
        {"analysis": analysis, "reframing": reframing, "pattern": pattern, "values": values},
        ensure_ascii=False,
        indent=2,
    )
    user_message = f"""4개 에이전트의 분석 결과입니다. 종합 판단을 내려주세요.

{agent_results}"""

    raw = _call_solar(
        [
            {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        api_key=api_key,
        temperature=0.5,
    )
    return _parse_json(raw)


def run_pipeline(
    keep: list[str],
    hard: list[str],
    try_: list[str],
    current_priority: dict[str, int],
    history: list[dict],
    api_key: Optional[str] = None,
    rag_context: Optional[str] = None,
    user_id: Optional[str] = None,
    week: Optional[int] = None,
) -> dict:
    """
    멀티에이전트 파이프라인 실행:
      1단계) 분석 에이전트 (항상 먼저 실행)
      2단계) 리프레이밍 + 패턴 + 가치관 에이전트 (병렬, 분석 결과를 컨텍스트로 주입)
      3단계) 슈퍼바이저 에이전트 (Solar로 전체 종합 + 최종 판단)

    Returns:
        {
          "analysis": {...},
          "reframing": {...},
          "pattern": {...},
          "values": {...},
          "supervisor": {...},   # 슈퍼바이저 최종 판단
          "errors": {...}        # 실패한 에이전트만 포함
        }
    """
    key = api_key or os.environ.get("UPSTAGE_API_KEY")
    if not key:
        raise ValueError("API 키가 필요합니다.")

    errors = {}

    # ── 1단계: 분석 에이전트 먼저 실행 ──────────────────────────────
    print("[1/3] 분석 에이전트 실행 중...")
    try:
        analysis = run_analysis_agent(
            keep, hard, try_,
            api_key=key, rag_context=rag_context, user_id=user_id, week=week,
        )
    except Exception as e:
        errors["analysis"] = str(e)
        analysis = {}

    # 분석 결과를 다음 에이전트에 넘길 컨텍스트로 변환
    analysis_context = None
    if analysis:
        analysis_context = (
            f"분석 에이전트 결과: "
            f"주요 감정={analysis.get('emotion', {}).get('primary', '')}, "
            f"번아웃 신호={analysis.get('burnout_signal', '')}, "
            f"고민 도메인={analysis.get('context', {}).get('domain', '')}, "
            f"요약={analysis.get('summary', '')}"
        )
        if rag_context:
            analysis_context += f"\n{rag_context}"

    # ── 2단계: 나머지 3개 병렬 실행 (분석 결과 컨텍스트 주입) ────────
    print("[2/3] 리프레이밍·패턴·가치관 에이전트 병렬 실행 중...")
    common = dict(api_key=key, rag_context=analysis_context, user_id=user_id, week=week)

    tasks = {
        "reframing": lambda: run_reframing_agent(keep, hard, try_, **common),
        "pattern": lambda: run_pattern_agent(
            {"keep": keep, "hard": hard, "try": try_}, history, **common
        ),
        "values": lambda: run_values_agent(keep, hard, try_, current_priority, **common),
    }

    results = {"analysis": analysis}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                errors[name] = str(e)
                results[name] = {}
                print(f"[{name}] 에러: {e}")

    # ── 3단계: 슈퍼바이저 종합 ────────────────────────────────────────
    print("[3/3] 슈퍼바이저 종합 중...")
    try:
        results["supervisor"] = _supervisor_synthesize(
            results.get("analysis", {}),
            results.get("reframing", {}),
            results.get("pattern", {}),
            results.get("values", {}),
            api_key=key,
        )
    except Exception as e:
        errors["supervisor"] = str(e)

    if errors:
        results["errors"] = errors

    return results


if __name__ == "__main__":
    sample_keep = ["알람 없이 기상", "산책 30분"]
    sample_hard = ["유튜브 보다 비교돼서 우울해짐", "자소서 한 줄도 못 씀"]
    sample_try = ["음악 틀어놓고 공부해보기"]

    sample_priority = {
        "근무지": 6, "워라밸": 2, "도메인": 3,
        "직무": 4, "연봉": 1, "기업규모": 5, "안정성": 7,
    }

    sample_history = [
        {"week": 1, "keep": ["운동"], "hard": ["면접 떨어짐", "의욕 없음"], "try": ["독서"]},
    ]

    api_key = input("Upstage API 키를 입력하세요: ").strip()
    print()

    result = run_pipeline(
        sample_keep, sample_hard, sample_try,
        current_priority=sample_priority,
        history=sample_history,
        api_key=api_key,
        user_id="test_user_01",
        week=2,
    )

    print("\n===== 최종 결과 =====\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
