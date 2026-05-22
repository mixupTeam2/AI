import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, TypedDict
from langgraph.graph import StateGraph, END

from analysis_agent import run_analysis_agent
from reframing_agent import run_reframing_agent
from pattern_agent import run_pattern_agent
from values_agent import run_values_agent
from type_agent import run_type_agent

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL = "solar-pro3"
VALID_AGENTS = {"analysis", "reframing", "pattern", "values"}

PLAN_PROMPT = """당신은 취업준비생 케어 시스템의 오케스트레이터입니다.
KHT 회고를 읽고 어떤 에이전트를 어떤 순서로 실행할지 결정하세요.

사용 가능한 에이전트는 정확히 아래 4개뿐입니다. 이 외의 에이전트는 절대 사용하지 마세요:
- analysis: 감정·맥락 분류, 번아웃 신호 판단
- reframing: Hard 항목을 강점 언어로 전환, 즉각 피드백
- pattern: CareType 4축 점수 채점 (E/A, P/X, G/R, C/S) — 매주 실행
- values: 가치관 변화 감지, 우선순위 업데이트 제안

판단 기준:
- Hard 항목 감정 강도가 높으면 → reframing을 먼저
- 번아웃 위험이 보이면 → analysis + pattern 우선
- 가치관 혼란 신호가 있으면 → values 포함
- pattern은 매주 항상 실행 (CareType 점수 누적용)

반드시 아래 JSON만 반환하세요:
{
  "reasoning": "판단 근거 한 줄",
  "first_agents": ["analysis" | "reframing" | "pattern" | "values"],
  "second_agents": ["analysis" | "reframing" | "pattern" | "values"],
  "skip": []
}"""

ADJUST_PROMPT = """당신은 취업준비생 케어 시스템의 오케스트레이터입니다.
1차 에이전트 실행 결과를 보고 추가 조정이 필요한지 판단하세요.

사용 가능한 에이전트: analysis, reframing, pattern, values (이 외 절대 사용 금지)

반드시 아래 JSON만 반환하세요:
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


# ── LangGraph State ────────────────────────────────────────────────────

class AgentState(TypedDict):
    keep: list[str]
    hard: list[str]
    try_: list[str]
    current_priority: dict
    axis_scores_history: list
    api_key: str
    rag_context: Optional[str]
    user_id: Optional[str]
    week: Optional[int]
    plan: dict
    adjustment: dict
    context: Optional[str]
    executed_agents: list
    analysis: dict
    reframing: dict
    pattern: dict
    values: dict
    supervisor: dict
    care_type: dict
    errors: dict


# ── 공통 유틸 ──────────────────────────────────────────────────────────

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


def _run_agent(name: str, state: AgentState, context: Optional[str]) -> dict:
    common = dict(api_key=state["api_key"], rag_context=context, user_id=state["user_id"], week=state["week"])
    if name == "analysis":
        return run_analysis_agent(state["keep"], state["hard"], state["try_"], **common)
    if name == "reframing":
        return run_reframing_agent(state["keep"], state["hard"], state["try_"], **common)
    if name == "pattern":
        return run_pattern_agent(state["keep"], state["hard"], state["try_"], **common)
    if name == "values":
        return run_values_agent(state["keep"], state["hard"], state["try_"], state["current_priority"], **common)
    raise ValueError(f"알 수 없는 에이전트: {name}")


def _run_parallel(names: list[str], state: AgentState, context: Optional[str]) -> dict:
    results, errors = {}, {}
    if not names:
        return results
    with ThreadPoolExecutor(max_workers=len(names)) as executor:
        futures = {executor.submit(_run_agent, n, state, context): n for n in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                errors[name] = str(e)
                print(f"[{name}] 에러: {e}")
    if errors:
        results["_errors"] = errors
    return results


# ── LangGraph 노드 ─────────────────────────────────────────────────────

def plan_node(state: AgentState) -> dict:
    print("[1/5] 오케스트레이터 실행 계획 수립 중...")
    kht_summary = (
        f"Keep: {', '.join(state['keep']) or '없음'}\n"
        f"Hard: {', '.join(state['hard']) or '없음'}\n"
        f"Try: {', '.join(state['try_']) or '없음'}\n"
        f"누적 주차 수: {len(state['axis_scores_history'])}"
    )
    plan = _parse_json(_call_solar([
        {"role": "system", "content": PLAN_PROMPT},
        {"role": "user", "content": kht_summary},
    ], state["api_key"]))
    print(f"      → {plan['reasoning']}")
    print(f"      → 1차: {plan['first_agents']} / 2차: {plan['second_agents']}")
    return {"plan": plan, "context": state.get("rag_context"), "executed_agents": [], "errors": {}}


def run_first_node(state: AgentState) -> dict:
    first_agents = [a for a in state["plan"].get("first_agents", []) if a in VALID_AGENTS]
    print(f"[2/5] 1차 에이전트 실행 중: {first_agents}")
    results = _run_parallel(first_agents, state, state.get("context"))

    analysis = results.get("analysis", {})
    context = state.get("context")
    if analysis:
        analysis_ctx = (
            f"분석결과: 주요감정={analysis.get('emotion', {}).get('primary', '')}, "
            f"번아웃신호={analysis.get('burnout_signal', '')}, "
            f"도메인={analysis.get('context', {}).get('domain', '')}"
        )
        context = f"{analysis_ctx}\n{context}" if context else analysis_ctx

    return {
        **{k: v for k, v in results.items() if k in VALID_AGENTS},
        "context": context,
        "executed_agents": [k for k in results if k in VALID_AGENTS],
        "errors": results.get("_errors", {}),
    }


def adjust_node(state: AgentState) -> dict:
    print("[3/5] 오케스트레이터 조정 판단 중...")
    first_results = {k: state.get(k, {}) for k in state["executed_agents"] if k in VALID_AGENTS}
    adjustment = _parse_json(_call_solar([
        {"role": "system", "content": ADJUST_PROMPT},
        {"role": "user", "content": (
            f"1차 실행 결과:\n{json.dumps(first_results, ensure_ascii=False)}\n\n"
            f"2차 예정 에이전트: {state['plan'].get('second_agents', [])}"
        )},
    ], state["api_key"]))
    print(f"      → {adjustment['reasoning']}")

    context = state.get("context")
    if adjustment.get("context_injection"):
        context = f"{adjustment['context_injection']}\n{context or ''}"

    return {"adjustment": adjustment, "context": context}


def run_second_node(state: AgentState) -> dict:
    second_agents = list(dict.fromkeys(
        state["plan"].get("second_agents", []) + state["adjustment"].get("add_agents", [])
    ))
    second_agents = [a for a in second_agents if a in VALID_AGENTS and a not in state["executed_agents"]]
    print(f"[4/5] 2차 에이전트 실행 중: {second_agents}")
    results = _run_parallel(second_agents, state, state.get("context"))
    return {
        **{k: v for k, v in results.items() if k in VALID_AGENTS},
        "executed_agents": state["executed_agents"] + [k for k in results if k in VALID_AGENTS],
        "errors": {**state.get("errors", {}), **results.get("_errors", {})},
    }


def synthesize_node(state: AgentState) -> dict:
    print("[5/5] 오케스트레이터 최종 메시지 생성 중...")
    all_results = {k: state.get(k, {}) for k in VALID_AGENTS}
    supervisor = _parse_json(_call_solar([
        {"role": "system", "content": SYNTHESIZE_PROMPT},
        {"role": "user", "content": f"에이전트 실행 결과:\n{json.dumps(all_results, ensure_ascii=False)}"},
    ], state["api_key"]))
    return {"supervisor": supervisor}


def type_node(state: AgentState) -> dict:
    print("[CareType] 10주 누적 → CareType 생성 중...")
    all_scores = state["axis_scores_history"] + [state.get("pattern", {})]
    care_type = run_type_agent(all_scores, api_key=state["api_key"], user_id=state["user_id"])
    return {"care_type": care_type}


# ── 조건 엣지 ──────────────────────────────────────────────────────────

def route_after_adjust(state: AgentState) -> str:
    second = list(dict.fromkeys(
        state["plan"].get("second_agents", []) + state["adjustment"].get("add_agents", [])
    ))
    second = [a for a in second if a in VALID_AGENTS and a not in state["executed_agents"]]
    return "run_second" if second else "synthesize"


def route_after_synthesize(state: AgentState) -> str:
    all_scores = state["axis_scores_history"] + [state.get("pattern", {})]
    return "type" if len(all_scores) >= 10 else END


# ── 그래프 빌드 ────────────────────────────────────────────────────────

def _build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("plan", plan_node)
    workflow.add_node("run_first", run_first_node)
    workflow.add_node("adjust", adjust_node)
    workflow.add_node("run_second", run_second_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("type", type_node)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "run_first")
    workflow.add_edge("run_first", "adjust")
    workflow.add_conditional_edges("adjust", route_after_adjust, {
        "run_second": "run_second",
        "synthesize": "synthesize",
    })
    workflow.add_edge("run_second", "synthesize")
    workflow.add_conditional_edges("synthesize", route_after_synthesize, {
        "type": "type",
        END: END,
    })
    workflow.add_edge("type", END)
    return workflow.compile()


_graph = _build_graph()


# ── 퍼블릭 API ────────────────────────────────────────────────────────

def run_pipeline(
    keep: list[str],
    hard: list[str],
    try_: list[str],
    current_priority: dict,
    axis_scores_history: list,
    api_key: Optional[str] = None,
    rag_context: Optional[str] = None,
    user_id: Optional[str] = None,
    week: Optional[int] = None,
) -> dict:
    key = api_key or os.environ.get("UPSTAGE_API_KEY")
    if not key:
        raise ValueError("API 키가 필요합니다.")

    initial_state: AgentState = {
        "keep": keep, "hard": hard, "try_": try_,
        "current_priority": current_priority,
        "axis_scores_history": axis_scores_history,
        "api_key": key, "rag_context": rag_context,
        "user_id": user_id, "week": week,
        "plan": {}, "adjustment": {}, "context": rag_context,
        "executed_agents": [],
        "analysis": {}, "reframing": {}, "pattern": {},
        "values": {}, "supervisor": {}, "care_type": {}, "errors": {},
    }

    final_state = _graph.invoke(initial_state)

    result = {k: final_state.get(k, {}) for k in ["analysis", "reframing", "pattern", "values", "supervisor"]}
    result["orchestrator_plan"] = final_state.get("plan", {})
    result["orchestrator_adjustment"] = final_state.get("adjustment", {})

    if final_state.get("care_type"):
        result["care_type"] = final_state["care_type"]
    else:
        all_scores = axis_scores_history + [final_state.get("pattern", {})]
        result["care_type"] = {
            "is_complete": False,
            "weeks_accumulated": len(all_scores),
            "weeks_remaining": max(0, 10 - len(all_scores)),
        }
    if final_state.get("errors"):
        result["errors"] = final_state["errors"]
    return result


if __name__ == "__main__":
    sample_keep = ["알람 없이 기상", "산책 30분"]
    sample_hard = ["유튜브 보다 비교돼서 우울해짐", "자소서 한 줄도 못 씀"]
    sample_try = ["음악 틀어놓고 공부해보기"]
    sample_priority = {
        "근무지": 6, "워라밸": 2, "도메인": 3,
        "직무": 4, "연봉": 1, "기업규모": 5, "안정성": 7,
    }

    api_key = input("Upstage API 키를 입력하세요: ").strip()
    print()
    result = run_pipeline(
        sample_keep, sample_hard, sample_try,
        current_priority=sample_priority,
        axis_scores_history=[],
        api_key=api_key,
        user_id="test_user_01",
        week=1,
    )
    print("\n===== 최종 결과 =====\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
