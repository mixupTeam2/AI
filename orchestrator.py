import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import requests

from analysis_agent import run_analysis_agent
from graph.pipeline import save_pipeline_result
from pattern_agent import run_pattern_agent
from rag.retriever import build_graph_rag_context
from reframing_agent import run_reframing_agent
from type_agent import run_type_agent
from values_agent import run_values_agent

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL = "solar-pro3"

VALID_AGENTS = {"analysis", "reframing", "pattern", "values"}

PLAN_PROMPT = """당신은 취업준비생 케어 시스템의 오케스트레이터입니다.
KHT 회고와 Graph RAG 문맥을 보고 어떤 에이전트를 어떤 순서로 실행할지 결정하세요.

사용 가능한 에이전트:
- analysis: 감정, 맥락, 번아웃/회복 신호 분석
- reframing: Hard 항목을 강점 언어로 전환
- pattern: CareType 4축 점수 채점(E/B, C/D, S/I, G/P)
- values: 가치관 변화 감지 및 우선순위 업데이트 제안

규칙:
- pattern은 CareType 누적을 위해 매주 실행하는 편이 좋습니다.
- first_agents와 second_agents에는 위 4개 이름만 넣으세요.
- 반드시 JSON만 반환하세요.

{
  "reasoning": "판단 근거 한 줄",
  "first_agents": ["analysis", "pattern"],
  "second_agents": ["reframing", "values"],
  "skip": []
}"""

ADJUST_PROMPT = """당신은 취업준비생 케어 시스템의 오케스트레이터입니다.
1차 에이전트 결과를 보고 2차 실행 계획을 조정하세요.

사용 가능한 에이전트는 analysis, reframing, pattern, values 뿐입니다.
이미 실행된 에이전트는 다시 추가하지 마세요.
반드시 JSON만 반환하세요.

{
  "reasoning": "조정 근거 한 줄",
  "add_agents": [],
  "context_injection": "다음 에이전트에게 전달할 핵심 문맥. 없으면 null"
}"""

SYNTHESIZE_PROMPT = """당신은 취업준비생 케어 시스템의 최종 슈퍼바이저입니다.
모든 에이전트 결과를 종합해 사용자에게 전달할 최종 케어 메시지를 만드세요.

반드시 JSON만 반환하세요.

{
  "care_point": "이번 주 가장 중요한 케어 포인트 한 줄",
  "main_message": "사용자에게 전달할 따뜻하고 구체적인 메시지 2~3문장",
  "action": "지금 바로 시도할 수 있는 행동 1개",
  "next_week_focus": "다음 주 집중 방향 한 줄",
  "orchestrator_log": "어떤 판단을 했는지 한 줄 요약"
}"""


def _call_solar(messages: list[dict[str, str]], api_key: str, temperature: float = 0.3) -> str:
    response = requests.post(
        SOLAR_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": SOLAR_MODEL, "messages": messages, "temperature": temperature},
        timeout=45,
    )
    if not response.ok:
        print("Solar error:", response.status_code, response.text)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _parse_json(raw: str) -> dict[str, Any]:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def _safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _agent_names(values: Any) -> list[str]:
    return [name for name in _safe_list(values) if name in VALID_AGENTS]


def _run_agent(
    name: str,
    keep: list[str],
    hard: list[str],
    try_: list[str],
    current_priority: dict[str, int],
    api_key: str,
    context: Optional[str] = None,
    user_id: Optional[str] = None,
    week: Optional[int] = None,
) -> dict[str, Any]:
    common = dict(api_key=api_key, rag_context=context, user_id=user_id, week=week)
    if name == "analysis":
        return run_analysis_agent(keep, hard, try_, **common)
    if name == "reframing":
        return run_reframing_agent(keep, hard, try_, **common)
    if name == "pattern":
        return run_pattern_agent(keep, hard, try_, **common)
    if name == "values":
        return run_values_agent(keep, hard, try_, current_priority, **common)
    raise ValueError(f"Unknown agent: {name}")


def _run_parallel(
    agent_names: list[str],
    context: Optional[str],
    keep: list[str],
    hard: list[str],
    try_: list[str],
    current_priority: dict[str, int],
    api_key: str,
    user_id: Optional[str],
    week: Optional[int],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    if not agent_names:
        return results

    with ThreadPoolExecutor(max_workers=len(agent_names)) as executor:
        futures = {
            executor.submit(
                _run_agent,
                name,
                keep,
                hard,
                try_,
                current_priority,
                api_key,
                context,
                user_id,
                week,
            ): name
            for name in agent_names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                print(f"[{name}] error: {exc}")
                results[name] = {"error": str(exc)}
    return results


def _build_ui_summary(results: dict[str, Any]) -> dict[str, Any]:
    analysis = results.get("analysis") or {}
    reframing = results.get("reframing") or {}
    pattern = results.get("pattern") or {}
    values = results.get("values") or {}
    supervisor = results.get("supervisor") or {}
    care_type = results.get("care_type") or {}

    emotion = analysis.get("emotion") or {}
    reframing_cards = []
    for item in _safe_list(reframing.get("reframing")):
        if isinstance(item, dict):
            reframing_cards.append(
                {
                    "original": item.get("original"),
                    "reframed": item.get("reframed"),
                    "strength_keywords": item.get("strength_keywords") or [],
                }
            )

    axis_scores = {
        key: pattern.get(key)
        for key in ("axis1", "axis2", "axis3", "axis4")
        if isinstance(pattern.get(key), dict)
    }

    return {
        "main_message": supervisor.get("main_message"),
        "care_point": supervisor.get("care_point"),
        "action": supervisor.get("action"),
        "next_week_focus": supervisor.get("next_week_focus"),
        "emotion": emotion.get("primary"),
        "emotion_intensity": emotion.get("intensity"),
        "burnout_signal": analysis.get("burnout_signal") or reframing.get("burnout_signal"),
        "recovery_signal": analysis.get("recovery_signal"),
        "analysis_summary": analysis.get("summary"),
        "value_change_detected": values.get("change_detected"),
        "changed_values": values.get("changed_values") or [],
        "value_update_message": values.get("update_message"),
        "reframing_cards": reframing_cards,
        "axis_scores": axis_scores,
        "care_type": care_type,
    }


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
) -> dict[str, Any]:
    key = api_key or os.environ.get("SOLAR_API_KEY") or os.environ.get("UPSTAGE_API_KEY")
    if not key:
        raise ValueError("SOLAR_API_KEY or UPSTAGE_API_KEY is required.")

    errors: dict[str, str] = {}
    current_week = {"keep": keep, "hard": hard, "try": try_}

    graph_rag_context = rag_context
    graph_rag_used = bool(graph_rag_context)
    if graph_rag_context is None and user_id:
        try:
            graph_rag_context = build_graph_rag_context(
                user_id=user_id,
                current_week=current_week,
                top_k=3,
                history_limit=5,
            )
            graph_rag_used = bool(graph_rag_context)
        except Exception as exc:
            errors["graph_rag"] = str(exc)

    kht_summary = f"""KHT report:
Keep: {', '.join(keep) or 'none'}
Hard: {', '.join(hard) or 'none'}
Try: {', '.join(try_) or 'none'}
Accumulated axis-score weeks: {len(axis_scores_history)}

{graph_rag_context or ''}"""

    print("[1/5] planning agent execution...")
    try:
        plan = _parse_json(
            _call_solar(
                [
                    {"role": "system", "content": PLAN_PROMPT},
                    {"role": "user", "content": kht_summary},
                ],
                key,
            )
        )
    except Exception as exc:
        errors["plan"] = str(exc)
        plan = {
            "reasoning": "fallback: run all core agents",
            "first_agents": ["analysis", "pattern"],
            "second_agents": ["reframing", "values"],
            "skip": [],
        }

    results: dict[str, Any] = {}
    context = graph_rag_context

    first_agents = _agent_names(plan.get("first_agents"))
    print(f"[2/5] first agents: {first_agents}")
    first_results = _run_parallel(
        first_agents,
        context,
        keep,
        hard,
        try_,
        current_priority,
        key,
        user_id,
        week,
    )
    results.update(first_results)

    print("[3/5] adjusting second pass...")
    try:
        adjustment = _parse_json(
            _call_solar(
                [
                    {"role": "system", "content": ADJUST_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"First results:\n{json.dumps(first_results, ensure_ascii=False)}\n\n"
                            f"Planned second agents: {plan.get('second_agents', [])}"
                        ),
                    },
                ],
                key,
            )
        )
    except Exception as exc:
        errors["adjustment"] = str(exc)
        adjustment = {"reasoning": "fallback", "add_agents": [], "context_injection": None}

    planned_second = _agent_names(plan.get("second_agents"))
    added_second = _agent_names(adjustment.get("add_agents"))
    second_agents = list(dict.fromkeys(planned_second + added_second))
    second_agents = [name for name in second_agents if name not in results]

    if adjustment.get("context_injection"):
        context = f"{adjustment['context_injection']}\n{context or ''}"

    print(f"[4/5] second agents: {second_agents}")
    second_results = _run_parallel(
        second_agents,
        context,
        keep,
        hard,
        try_,
        current_priority,
        key,
        user_id,
        week,
    )
    results.update(second_results)

    print("[5/5] synthesizing final response...")
    try:
        results["supervisor"] = _parse_json(
            _call_solar(
                [
                    {"role": "system", "content": SYNTHESIZE_PROMPT},
                    {
                        "role": "user",
                        "content": f"Agent results:\n{json.dumps(results, ensure_ascii=False)}",
                    },
                ],
                key,
                temperature=0.5,
            )
        )
    except Exception as exc:
        errors["supervisor"] = str(exc)
        results["supervisor"] = {}

    if results.get("pattern") and not results["pattern"].get("error"):
        all_scores = axis_scores_history + [results["pattern"]]
        if len(all_scores) >= 10:
            try:
                results["care_type"] = run_type_agent(all_scores, api_key=key, user_id=user_id)
            except Exception as exc:
                errors["care_type"] = str(exc)
        else:
            results["care_type"] = {
                "is_complete": False,
                "weeks_accumulated": len(all_scores),
                "weeks_remaining": 10 - len(all_scores),
            }

    results["orchestrator_plan"] = plan
    results["orchestrator_adjustment"] = adjustment
    results["graph_rag_used"] = graph_rag_used
    results["ui_summary"] = _build_ui_summary(results)

    if user_id and week is not None:
        try:
            save_pipeline_result(
                user_id=user_id,
                week=week,
                keep=keep,
                hard=hard,
                try_=try_,
                current_priority=current_priority,
                results=results,
            )
            results["graph_saved"] = True
        except Exception as exc:
            errors["graph_save"] = str(exc)
            results["graph_saved"] = False

    if errors:
        results["errors"] = errors

    return results


if __name__ == "__main__":
    sample_keep = ["자소서 초안을 끝까지 작성했다", "스터디에서 모의면접 피드백을 받았다"]
    sample_hard = ["다른 사람 스펙을 보면서 위축됐다", "지원 직무가 정말 나와 맞는지 확신이 떨어졌다"]
    sample_try = ["SNS 사용 시간을 줄인다", "직무 공고 5개를 분석한다"]
    sample_priority = {"워라밸": 1, "직무": 2, "도메인": 3, "근무지": 4, "안정성": 5, "연봉": 6, "기업규모": 7}

    result = run_pipeline(
        sample_keep,
        sample_hard,
        sample_try,
        current_priority=sample_priority,
        axis_scores_history=[],
        user_id="demo_user_001",
        week=1,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
