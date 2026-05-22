from typing import Any

from graph.queries import get_recent_report_context, get_user_profile
from graph.schema import result_values, run_query


def find_similar_users(user_id: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Find similar users from values, emotions, care type, and tags."""
    result = run_query(
        """
        MATCH (target:User {user_id: $user_id})
        MATCH (other:User)
        WHERE other.user_id <> $user_id

        OPTIONAL MATCH (target)-[tv:PRIORITIZES]->(v:Value)<-[ov:PRIORITIZES]-(other)
        WITH target, other,
             count(DISTINCT v) AS shared_values,
             sum(abs(coalesce(tv.priority, 0) - coalesce(ov.priority, 0))) AS priority_diff

        OPTIONAL MATCH (target)-[:FEELS]->(e:Emotion)<-[:FEELS]-(other)
        WITH target, other, shared_values, priority_diff,
             count(DISTINCT e) AS shared_emotions

        OPTIONAL MATCH (target)-[:HAS_TYPE]->(c:CareType)<-[:HAS_TYPE]-(other)
        WITH target, other, shared_values, priority_diff, shared_emotions,
             count(DISTINCT c) AS shared_type

        OPTIONAL MATCH (target)-[:HAS_TAG]->(t:GraphTag)<-[:HAS_TAG]-(other)
        WITH other, shared_values, priority_diff, shared_emotions, shared_type,
             count(DISTINCT t) AS shared_tags

        WITH other,
             (
                 shared_values * 3
                 + shared_emotions * 2
                 + shared_type * 2
                 + shared_tags
                 - coalesce(priority_diff, 0) * 0.5
             ) AS similarity_score
        ORDER BY similarity_score DESC
        LIMIT $top_k

        OPTIONAL MATCH (other)-[feel:FEELS]->(e:Emotion)
        OPTIONAL MATCH (other)-[prior:PRIORITIZES]->(v:Value)
        OPTIONAL MATCH (other)-[:HAS_TYPE]->(ct:CareType)
        OPTIONAL MATCH (other)-[:WORRIES_ABOUT]->(con:Concern)

        RETURN
            other.user_id AS user_id,
            ct.code AS care_type,
            collect(DISTINCT {name: e.name, week: feel.week}) AS emotions,
            collect(DISTINCT {name: v.name, priority: prior.priority}) AS values,
            collect(DISTINCT con.name) AS concern_keywords,
            similarity_score
        """,
        {"user_id": user_id, "top_k": top_k},
    )

    recommendations = []
    for row in result_values(result):
        emotions = [item for item in (row[2] or []) if item.get("name")]
        emotions.sort(key=lambda item: item.get("week") or 0, reverse=True)

        values = [item for item in (row[3] or []) if item.get("name")]
        values.sort(key=lambda item: item.get("priority") or 999)

        recommendations.append(
            {
                "user_id": row[0],
                "care_type": row[1],
                "emotion": emotions[0]["name"] if emotions else None,
                "value_priority": [item["name"] for item in values],
                "concern_keywords": [item for item in (row[4] or []) if item],
                "similarity_score": round(float(row[5] or 0), 2),
            }
        )

    return recommendations


def _format_current_week(current_week: dict[str, Any] | None) -> list[str]:
    if not current_week:
        return []
    lines = []
    for key, label in [("keep", "Keep"), ("hard", "Hard"), ("try", "Try")]:
        values = current_week.get(key) or []
        if values:
            lines.append(f"- {label}: {', '.join(map(str, values))}")
    return lines


def build_graph_rag_context(
    user_id: str,
    current_week: dict[str, Any] | None = None,
    top_k: int = 3,
    history_limit: int = 5,
) -> str:
    """Build compact prompt context from Neo4j for all agents."""
    profile = get_user_profile(user_id)
    previous_reports = get_recent_report_context(user_id, limit=history_limit)
    similar_users = find_similar_users(user_id, top_k=top_k)

    sections: list[str] = ["[Graph RAG context]"]

    if profile:
        sections.append(
            "\nUser graph profile:\n"
            f"- latest_week: {profile.get('latest_week')}\n"
            f"- care_type: {profile.get('care_type') or 'unknown'}\n"
            f"- values: {', '.join(profile.get('value_priority') or []) or 'unknown'}\n"
            f"- concerns: {', '.join(profile.get('concerns') or []) or 'none'}"
        )

    current_lines = _format_current_week(current_week)
    if current_lines:
        sections.append("\nCurrent KHT input:\n" + "\n".join(current_lines))

    if previous_reports:
        lines = []
        for report in previous_reports:
            pieces = [
                f"week {report.get('week')}",
                f"emotion={report.get('emotion') or 'unknown'}",
                f"burnout={report.get('burnout_signal') or 'unknown'}",
                f"care_type={report.get('care_type') or 'unknown'}",
            ]
            if report.get("summary"):
                pieces.append(f"summary={report['summary']}")
            if report.get("reframing"):
                pieces.append(f"reframing={report['reframing']}")
            if report.get("value_change_message"):
                pieces.append(f"value_change={report['value_change_message']}")
            lines.append("- " + " | ".join(pieces))
        sections.append("\nRecent user history:\n" + "\n".join(lines))

    if similar_users:
        lines = []
        for user in similar_users:
            lines.append(
                "- "
                f"{user['user_id']} "
                f"(score={user['similarity_score']}, type={user.get('care_type') or 'unknown'}, "
                f"emotion={user.get('emotion') or 'unknown'}, "
                f"values={', '.join(user.get('value_priority') or []) or 'unknown'}, "
                f"concerns={', '.join(user.get('concern_keywords') or []) or 'none'})"
            )
        sections.append("\nSimilar users:\n" + "\n".join(lines))

    if len(sections) == 1:
        return ""
    sections.append(
        "\nUse this graph context only as supporting evidence. Do not expose user IDs in final coaching copy."
    )
    return "\n".join(sections)


def build_agent_rag_context(
    agent_name: str,
    user_id: str | None,
    current_week: dict[str, Any] | None = None,
    fallback: str | None = None,
) -> str | None:
    """Direct-agent helper: fetch graph context only when caller did not pass one."""
    if fallback:
        return fallback
    if not user_id:
        return None
    try:
        context = build_graph_rag_context(user_id=user_id, current_week=current_week)
    except Exception:
        return None
    if not context:
        return None
    return f"[Agent: {agent_name}]\n{context}"


if __name__ == "__main__":
    result = find_similar_users("user_001")
    print("\nuser_001 recommendations:")
    for item in result:
        print(
            f"  - {item['user_id']} | score={item['similarity_score']} "
            f"| emotion={item['emotion']} | values={item['value_priority']}"
        )
