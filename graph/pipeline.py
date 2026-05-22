import json
from pathlib import Path
from typing import Any

from graph.schema import run_query


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _priority_list(value_priority: Any) -> list[str]:
    if isinstance(value_priority, dict):
        return [
            value
            for value, _ in sorted(value_priority.items(), key=lambda item: item[1])
        ]
    return [str(value) for value in _as_list(value_priority) if value]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _dedupe_strings(values: list[Any], limit: int = 40) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def save_user_report(report: dict[str, Any]) -> None:
    """Save one weekly report plus searchable graph facts into Neo4j."""
    user_id = report["user_id"]
    week = report["week"]
    value_priority = _priority_list(report.get("value_priority"))
    concerns = _dedupe_strings(_as_list(report.get("concern_keywords")))
    strengths = _dedupe_strings(_as_list(report.get("strength_keywords")))
    graph_tags = _dedupe_strings(_as_list(report.get("graph_tags")))

    run_query(
        """
        MERGE (u:User {user_id: $user_id})
        SET u.latest_week = CASE
            WHEN u.latest_week IS NULL OR u.latest_week < $week THEN $week
            ELSE u.latest_week
        END
        WITH u
        MERGE (r:WeeklyReport {user_id: $user_id, week: $week})
        SET
            r.keep = $keep,
            r.hard = $hard,
            r.try_next = $try_next,
            r.summary = $summary,
            r.emotion = $emotion,
            r.emotion_signal = $emotion_signal,
            r.burnout_signal = $burnout_signal,
            r.recovery_signal = $recovery_signal,
            r.reframing = $reframing,
            r.value_changed = $value_changed,
            r.value_change_message = $value_change_message,
            r.care_type = $care_type,
            r.axis_scores_json = $axis_scores_json,
            r.type_result_json = $type_result_json,
            r.supervisor_json = $supervisor_json,
            r.agent_json = $agent_json,
            r.ui_summary_json = $ui_summary_json,
            r.graph_tags = $graph_tags
        MERGE (u)-[:SUBMITTED]->(r)
        """,
        {
            "user_id": user_id,
            "week": week,
            "keep": _as_list(report.get("keep")),
            "hard": _as_list(report.get("hard")),
            "try_next": _as_list(report.get("try_next") or report.get("try")),
            "summary": report.get("summary"),
            "emotion": report.get("emotion"),
            "emotion_signal": report.get("emotion_signal"),
            "burnout_signal": report.get("burnout_signal"),
            "recovery_signal": report.get("recovery_signal"),
            "reframing": report.get("reframing"),
            "value_changed": bool(report.get("value_changed", False)),
            "value_change_message": report.get("value_change_message"),
            "care_type": report.get("care_type"),
            "axis_scores_json": _json(report.get("axis_scores")),
            "type_result_json": _json(report.get("type_result")),
            "supervisor_json": _json(report.get("supervisor", {})),
            "agent_json": _json(report.get("agent_results", {})),
            "ui_summary_json": _json(report.get("ui_summary", {})),
            "graph_tags": graph_tags,
        },
    )

    if report.get("care_type"):
        run_query(
            """
            MATCH (u:User {user_id: $user_id})
            MERGE (c:CareType {code: $code})
            MERGE (u)-[r:HAS_TYPE]->(c)
            SET r.week = $week
            """,
            {"code": report["care_type"], "user_id": user_id, "week": week},
        )

    if report.get("emotion"):
        run_query(
            """
            MATCH (u:User {user_id: $user_id})
            MERGE (e:Emotion {name: $emotion})
            MERGE (u)-[r:FEELS]->(e)
            SET
                r.week = $week,
                r.signal = $emotion_signal,
                r.burnout_signal = $burnout_signal,
                r.recovery_signal = $recovery_signal
            """,
            {
                "emotion": report.get("emotion"),
                "user_id": user_id,
                "week": week,
                "emotion_signal": report.get("emotion_signal"),
                "burnout_signal": report.get("burnout_signal"),
                "recovery_signal": report.get("recovery_signal"),
            },
        )

    for idx, value in enumerate(value_priority):
        run_query(
            """
            MATCH (u:User {user_id: $user_id})
            MERGE (v:Value {name: $value})
            MERGE (u)-[r:PRIORITIZES]->(v)
            SET r.priority = $priority, r.week = $week
            """,
            {
                "value": value,
                "user_id": user_id,
                "priority": idx + 1,
                "week": week,
            },
        )

    for concern in concerns:
        run_query(
            """
            MATCH (u:User {user_id: $user_id})
            MERGE (c:Concern {name: $concern})
            MERGE (u)-[r:WORRIES_ABOUT]->(c)
            SET r.week = $week
            """,
            {"concern": concern, "user_id": user_id, "week": week},
        )

    for strength in strengths:
        run_query(
            """
            MATCH (u:User {user_id: $user_id})
            MERGE (s:Strength {name: $strength})
            MERGE (u)-[r:SHOWS_STRENGTH]->(s)
            SET r.week = $week
            """,
            {"strength": strength, "user_id": user_id, "week": week},
        )

    for tag in graph_tags:
        run_query(
            """
            MATCH (u:User {user_id: $user_id})
            MERGE (t:GraphTag {name: $tag})
            MERGE (u)-[r:HAS_TAG]->(t)
            SET r.week = $week
            """,
            {"tag": tag, "user_id": user_id, "week": week},
        )


def _collect_strengths(reframing: dict[str, Any]) -> list[str]:
    strengths = []
    for item in _as_list(reframing.get("reframing")):
        if isinstance(item, dict):
            strengths.extend(_as_list(item.get("strength_keywords")))
    strengths.extend(_as_list(reframing.get("strength_keywords")))
    return _dedupe_strings(strengths)


def _collect_graph_tags(results: dict[str, Any]) -> list[str]:
    analysis = results.get("analysis") or {}
    reframing = results.get("reframing") or {}
    pattern = results.get("pattern") or {}
    values = results.get("values") or {}
    supervisor = results.get("supervisor") or {}
    care_type = results.get("care_type") or {}

    tags: list[Any] = []
    tags.extend(_as_list(analysis.get("graph_tags")))
    tags.extend(_as_list(reframing.get("graph_keywords")))
    tags.extend(_as_list(values.get("graph_tags")))
    tags.extend(_as_list(values.get("changed_values")))
    tags.append((analysis.get("emotion") or {}).get("primary"))
    tags.append((analysis.get("context") or {}).get("domain"))
    tags.append(supervisor.get("care_point"))
    tags.append(care_type.get("code"))

    for axis_key in ("axis1", "axis2", "axis3", "axis4"):
        axis = pattern.get(axis_key) or {}
        tags.extend(_as_list(axis.get("detected")))

    return _dedupe_strings(tags)


def _collect_concerns(results: dict[str, Any]) -> list[str]:
    analysis = results.get("analysis") or {}
    values = results.get("values") or {}
    supervisor = results.get("supervisor") or {}
    context = analysis.get("context") or {}
    return _dedupe_strings(
        [
            context.get("domain"),
            context.get("trigger"),
            supervisor.get("care_point"),
            *(_as_list(values.get("changed_values"))),
        ]
    )


def save_pipeline_result(
    user_id: str,
    week: int,
    keep: list[str],
    hard: list[str],
    try_: list[str],
    current_priority: dict[str, int] | list[str],
    results: dict[str, Any],
) -> None:
    """Convert multi-agent output into graph facts and persist it."""
    analysis = results.get("analysis") or {}
    reframing = results.get("reframing") or {}
    values = results.get("values") or {}
    pattern = results.get("pattern") or {}
    supervisor = results.get("supervisor") or {}
    care_type = results.get("care_type") or {}
    ui_summary = results.get("ui_summary") or {}

    emotion = analysis.get("emotion") or {}
    report = {
        "user_id": user_id,
        "week": week,
        "care_type": care_type.get("code") or supervisor.get("care_type"),
        "keep": keep,
        "hard": hard,
        "try_next": try_,
        "emotion": emotion.get("primary"),
        "emotion_signal": emotion.get("intensity"),
        "burnout_signal": analysis.get("burnout_signal") or reframing.get("burnout_signal"),
        "recovery_signal": analysis.get("recovery_signal"),
        "concern_keywords": _collect_concerns(results),
        "value_priority": current_priority,
        "value_changed": bool(values.get("change_detected")),
        "value_change_message": values.get("update_message"),
        "reframing": reframing.get("overall_message"),
        "strength_keywords": _collect_strengths(reframing),
        "summary": analysis.get("summary") or supervisor.get("main_message"),
        "axis_scores": pattern,
        "type_result": care_type,
        "graph_tags": _collect_graph_tags(results),
        "supervisor": supervisor,
        "ui_summary": ui_summary,
        "agent_results": {
            "analysis": analysis,
            "reframing": reframing,
            "pattern": pattern,
            "values": values,
            "care_type": care_type,
        },
    }
    save_user_report(report)


def load_dummy_data() -> None:
    dummy_path = Path(__file__).resolve().parents[1] / "routers" / "dummy_data.json"
    with dummy_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Saving {len(data)} dummy reports...")
    for report in data:
        save_user_report(report)
    print("\nDummy data saved.")


if __name__ == "__main__":
    load_dummy_data()
