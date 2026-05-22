import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from analysis_agent import run_analysis_agent
from reframing_agent import run_reframing_agent
from pattern_agent import run_pattern_agent
from values_agent import run_values_agent
from type_agent import run_type_agent

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL = "solar-pro3"

AGENT_REGISTRY = {
    "analysis": "감정·맥락 분류, 번아웃 신호 판단",
    "reframing": "Hard 항목을 강점 언어로 전환, 즉각 피드백",
    "pattern": "CareType 4축 점수 채점 (E/B, C/D, S/I, G/P)",
    "values": "가치관 변화 감지, 우선순위 업데이트 제안",
}

VALID_AGENTS = {"analysis", "reframing", "pattern", "values"}

PLAN_PROMPT = """당신은 취업준비생 케어 시스템의 오케스트레이터입니다.
KHT 회고를 읽고 어떤 에이전트를 어떤 순서로 실행할지 결정하세요.

사용 가능한 에이전트는 정확히 아래 4개뿐입니다. 이 외의 에이전트는 절대 사용하지 마세요:
- analysis: 감정·맥락 분류, 번아웃 신호 판단
- reframing: Hard 항목을 강점 언어로 전환, 즉각 피드백
- pattern: CareType 4축 점수 채점 (E/B, C/D, S/I, G/P) — 매주 실행
- values: 가치관 변화 감지, 우선순위 업데이트 제안

판단 기준:
- Hard 항목 감정 강도가 높으면 → reframing을 먼저
- 번아웃 위험이 보이면 → analysis + pattern 우선
- 가치관 혼란 신호가 있으면 → values 포함
- pattern은 매주 항상 실행 (CareType 점수 누적용)

반드시 아래 JSON만 반환하세요. first_agents와 second_agents에는 위 4개 중에서만 선택하세요:
{
  "reasoning": "판단 근거 한 줄",
  "first_agents": ["analysis" | "reframing" | "pattern" | "values"],
  "second_agents": ["analysis" | "reframing" | "pattern" | "values"],
  "skip": ["이번 주 불필요한 에이전트 (없으면 빈 리스트)"]
}"""

ADJUST_PROMPT = """당신은 취업준비생 케어 시스템의 오케스트레이터입니다.
1차 에이전트 실행 결과를 보고 추가 조정이 필요한지 판단하세요.

사용 가능한 에이전트는 정확히 아래 4개뿐입니다. 이 외의 에이전트는 절대 추가하지 마세요:
- analysis, reframing, pattern, values

반드시 아래 JSON만 반환하세요. add_agents에는 위 4개 중에서만 선택하세요:
{
  "reasoning": "판단 근거 한 줄",
  "add_agents": ["analysis" | "reframing" | "pattern" | "values"],
  "context_injection": "다음 에이전트에 전달할 핵심 컨텍스트 한 줄 (없으면 null)"
}"""

SYNTHESIZE_PROMPT = """당신은 취업준비생 케어 시스템의 오케스트레이터입니다.
모든 에이전트 결과를 종합해 유저에게 전달할 최종 케어 메시지를 만드세요.

반드시 아래 JSON만 반환하세요:
{
  "care_point": "이번 주 가장 시급한 케어 포인트 한 줄",
  "main_message": "유저에게 전달할 핵심 메시지 (2~3문장, 따뜻하고 구체적으로)",
  "action": "지금 당장 시도할 수 있는 행동 1가지",
  "next_week_focus": "다음 주 집중할 방향 한 줄",
  "orchestrator_log": "오케스트레이터가 어떤 판단을 했는지 한 줄 요약"
}"""


def _call_solar(messages: list[dict], api_key: str, temperature: float = 0.3) -> str:
    response = requests.post(
        SOLAR_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": SOLAR_MODEL, "messages": messages, "temperature": temperature},
        timeout=30,
    )
    if not response.ok:
        print("Solar 에러:", response.status_code, response.text)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _parse_json(raw: str) -> dict:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def _run_agent(name: str, keep, hard, try_, current_priority, api_key, context=None, user_id=None, week=None) -> dict:
    common = dict(api_key=api_key, rag_context=context, user_id=user_id, week=week)
    if name == "analysis":
        return run_analysis_agent(keep, hard, try_, **common)
    if name == "reframing":
        return run_reframing_agent(keep, hard, try_, **common)
    if name == "pattern":
        return run_pattern_agent(keep, hard, try_, **common)
    if name == "values":
        return run_values_agent(keep, hard, try_, current_priority, **common)
    raise ValueError(f"알 수 없는 에이전트: {name}")


def _run_parallel(agent_names: list[str], context: Optional[str], keep, hard, try_, current_priority, api_key, user_id, week) -> dict:
    results = {}
    if not agent_names:
        return results
    with ThreadPoolExecutor(max_workers=len(agent_names)) as executor:
        futures = {
            executor.submit(_run_agent, name, keep, hard, try_, current_priority, api_key, context, user_id, week): name
            for name in agent_names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                print(f"[{name}] 에러: {e}")
                results[name] = {"error": str(e)}
    return results


def run_pipeline(
    keep: list[str],
    hard: list[str],
    try_: list[str],
    current_priority: dict[str, int],
    axis_scores_history: list[dict],
    api_key: Optional[str] = None,
    rag_context: Optional[str] = None,
    user_id: Optional[str] = None,
    week: Optional[int] = None,
) -> dict:
    """
    Solar가 직접 판단하는 에이전틱 오케스트레이터:
      1) Solar가 KHT 읽고 실행 계획 결정
      2) 1차 에이전트 실행
      3) Solar가 결과 보고 추가 조정 판단
      4) 2차 에이전트 실행 (필요 시)
      5) Solar가 전체 종합
      6) 10주 누적 시 type_agent 자동 실행

    Args:
        axis_scores_history: 이전 주차 pattern_agent 결과 리스트
                             [{"axis1": {"score": 1}, ...}, ...]
    """
    key = api_key or os.environ.get("UPSTAGE_API_KEY")
    if not key:
        raise ValueError("API 키가 필요합니다.")

    kht_summary = f"""KHT 회고:
Keep: {', '.join(keep) or '없음'}
Hard: {', '.join(hard) or '없음'}
Try: {', '.join(try_) or '없음'}
누적 주차 수: {len(axis_scores_history)}"""

    # ── 1단계: Solar가 실행 계획 결정 ─────────────────────────────────
    print("[1/4] 오케스트레이터 실행 계획 수립 중...")
    plan = _parse_json(_call_solar([
        {"role": "system", "content": PLAN_PROMPT},
        {"role": "user", "content": kht_summary},
    ], key))
    print(f"      → {plan['reasoning']}")
    print(f"      → 1차: {plan['first_agents']} / 2차: {plan['second_agents']} / 스킵: {plan['skip']}")

    results = {}
    context = rag_context

    # ── 2단계: 1차 에이전트 실행 ──────────────────────────────────────
    first_agents = [a for a in plan["first_agents"] if a in VALID_AGENTS]
    print(f"[2/4] 1차 에이전트 실행 중: {first_agents}")
    first_results = _run_parallel(
        first_agents, context,
        keep, hard, try_, current_priority, key, user_id, week,
    )
    results.update(first_results)

    # ── 3단계: Solar가 중간 결과 보고 조정 판단 ───────────────────────
    print("[3/4] 오케스트레이터 조정 판단 중...")
    first_summary = json.dumps(first_results, ensure_ascii=False)
    adjustment = _parse_json(_call_solar([
        {"role": "system", "content": ADJUST_PROMPT},
        {"role": "user", "content": f"1차 실행 결과:\n{first_summary}\n\n2차 예정 에이전트: {plan['second_agents']}"},
    ], key))
    print(f"      → {adjustment['reasoning']}")

    # 조정된 2차 에이전트 목록 (원래 계획 + 추가 에이전트, 중복·유효하지 않은 항목 제거)
    second_agents = list(dict.fromkeys(plan["second_agents"] + adjustment.get("add_agents", [])))
    second_agents = [a for a in second_agents if a in VALID_AGENTS and a not in results]

    # 컨텍스트 업데이트
    if adjustment.get("context_injection"):
        context = f"{adjustment['context_injection']}\n{context or ''}"

    # ── 4단계: 2차 에이전트 실행 ──────────────────────────────────────
    if second_agents:
        print(f"[4/4] 2차 에이전트 실행 중: {second_agents}")
        second_results = _run_parallel(
            second_agents, context,
            keep, hard, try_, current_priority, key, user_id, week,
        )
        results.update(second_results)
    else:
        print("[4/4] 추가 에이전트 없음, 종합 단계로 이동")

    # ── 5단계: Solar가 전체 종합 ──────────────────────────────────────
    print("[종합] 오케스트레이터 최종 메시지 생성 중...")
    all_results_text = json.dumps(results, ensure_ascii=False)
    supervisor = _parse_json(_call_solar([
        {"role": "system", "content": SYNTHESIZE_PROMPT},
        {"role": "user", "content": f"에이전트 실행 결과:\n{all_results_text}"},
    ], key))
    results["supervisor"] = supervisor
    results["orchestrator_plan"] = plan

    # ── 6단계: 10주 누적 시 CareType 생성 ────────────────────────────
    if results.get("pattern") and not results["pattern"].get("error"):
        all_scores = axis_scores_history + [results["pattern"]]
        if len(all_scores) >= 10:
            print("[CareType] 10주 누적 달성 → CareType 생성 중...")
            results["care_type"] = run_type_agent(all_scores, api_key=key, user_id=user_id)
        else:
            results["care_type"] = {
                "is_complete": False,
                "weeks_accumulated": len(all_scores),
                "weeks_remaining": 10 - len(all_scores),
            }

    return results


if __name__ == "__main__":
    sample_keep = ["알람 없이 기상", "산책 30분"]
    sample_hard = ["유튜브 보다 비교돼서 우울해짐", "자소서 한 줄도 못 씀"]
    sample_try = ["음악 틀어놓고 공부해보기"]

    sample_priority = {
        "근무지": 6, "워라밸": 2, "도메인": 3,
        "직무": 4, "연봉": 1, "기업규모": 5, "안정성": 7,
    }

    # 이전 주차 pattern_agent 결과 (누적 축 점수)
    sample_axis_scores_history = [
        {"axis1": {"score": 1}, "axis2": {"score": 0}, "axis3": {"score": 2}, "axis4": {"score": 1}},
    ]

    api_key = input("Upstage API 키를 입력하세요: ").strip()
    print()

    result = run_pipeline(
        sample_keep, sample_hard, sample_try,
        current_priority=sample_priority,
        axis_scores_history=sample_axis_scores_history,
        api_key=api_key,
        user_id="test_user_01",
        week=2,
    )

    print("\n===== 최종 결과 =====\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
