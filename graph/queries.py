from typing import Any

from graph.schema import result_values, run_query


def get_current_value_priority(user_id: str) -> dict[str, int]:
    result = run_query(
        """
        MATCH (:User {user_id: $user_id})-[r:PRIORITIZES]->(v:Value)
        WITH v.name AS value, r.priority AS priority, r.week AS week
        ORDER BY week DESC, priority ASC
        WITH value, collect({priority: priority, week: week})[0] AS latest
        RETURN value, latest.priority AS priority
        ORDER BY priority ASC
        """,
        {"user_id": user_id},
    )
    return {
        row[0]: int(row[1])
        for row in result_values(result)
        if row[0] is not None and row[1] is not None
    }


def get_user_history(user_id: str, limit: int = 8) -> list[dict[str, Any]]:
    result = run_query(
        """
        MATCH (:User {user_id: $user_id})-[:SUBMITTED]->(r:WeeklyReport)
        RETURN r.week, r.keep, r.hard, r.try_next
        ORDER BY r.week DESC
        LIMIT $limit
        """,
        {"user_id": user_id, "limit": limit},
    )
    history = []
    for row in result_values(result):
        history.append(
            {
                "week": row[0],
                "keep": row[1] or [],
                "hard": row[2] or [],
                "try": row[3] or [],
            }
        )
    return list(reversed(history))


def get_axis_scores_history(user_id: str, limit: int = 12) -> list[dict[str, Any]]:
    result = run_query(
        """
        MATCH (:User {user_id: $user_id})-[:SUBMITTED]->(r:WeeklyReport)
        WHERE r.axis_scores_json IS NOT NULL
        RETURN r.week, r.axis_scores_json
        ORDER BY r.week DESC
        LIMIT $limit
        """,
        {"user_id": user_id, "limit": limit},
    )
    scores = []
    for row in result_values(result):
        import json

        try:
            parsed = json.loads(row[1]) if isinstance(row[1], str) else row[1]
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            parsed["week"] = row[0]
            scores.append(parsed)
    return list(reversed(scores))


def get_recent_report_context(user_id: str, limit: int = 5) -> list[dict[str, Any]]:
    result = run_query(
        """
        MATCH (:User {user_id: $user_id})-[:SUBMITTED]->(r:WeeklyReport)
        RETURN
            r.week,
            r.summary,
            r.emotion,
            r.burnout_signal,
            r.recovery_signal,
            r.reframing,
            r.value_change_message,
            r.graph_tags,
            r.axis_scores_json,
            r.care_type
        ORDER BY r.week DESC
        LIMIT $limit
        """,
        {"user_id": user_id, "limit": limit},
    )
    reports = []
    for row in result_values(result):
        reports.append(
            {
                "week": row[0],
                "summary": row[1],
                "emotion": row[2],
                "burnout_signal": row[3],
                "recovery_signal": row[4],
                "reframing": row[5],
                "value_change_message": row[6],
                "graph_tags": row[7] or [],
                "axis_scores_json": row[8],
                "care_type": row[9],
            }
        )
    return reports


def get_user_profile(user_id: str) -> dict[str, Any]:
    result = run_query(
        """
        MATCH (u:User {user_id: $user_id})
        OPTIONAL MATCH (u)-[p:PRIORITIZES]->(v:Value)
        WITH u, collect(DISTINCT {name: v.name, priority: p.priority}) AS values
        OPTIONAL MATCH (u)-[f:FEELS]->(e:Emotion)
        WITH u, values, collect(DISTINCT {name: e.name, week: f.week, signal: f.signal}) AS emotions
        OPTIONAL MATCH (u)-[:WORRIES_ABOUT]->(c:Concern)
        WITH u, values, emotions, collect(DISTINCT c.name) AS concerns
        OPTIONAL MATCH (u)-[:HAS_TYPE]->(ct:CareType)
        RETURN u.user_id, u.latest_week, ct.code, values, emotions, concerns
        """,
        {"user_id": user_id},
    )
    values = result_values(result)
    if not values:
        return {}

    row = values[0]
    priority = [
        item
        for item in (row[3] or [])
        if item.get("name") is not None and item.get("priority") is not None
    ]
    priority.sort(key=lambda item: item["priority"])

    emotions = [item for item in (row[4] or []) if item.get("name") is not None]
    emotions.sort(key=lambda item: item.get("week") or 0, reverse=True)

    return {
        "user_id": row[0],
        "latest_week": row[1],
        "care_type": row[2],
        "value_priority": [item["name"] for item in priority],
        "recent_emotions": emotions[:5],
        "concerns": [item for item in (row[5] or []) if item],
    }
