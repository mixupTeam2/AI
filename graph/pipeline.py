import json
import os
from graph.schema import run_query


def save_user_onboarding(user: dict):
    """온보딩 데이터 (스펙 + 가치관) Neo4j에 저장"""

    run_query("""
        MERGE (u:User {user_id: $user_id})
        SET u.gpa = $gpa,
            u.lab_experience = $lab_experience,
            u.latest_week = 0
    """, {
        "user_id": user["user_id"],
        "gpa": user["spec"].get("gpa", 0),
        "lab_experience": user["spec"].get("lab_experience", False),
    })

    for idx, value in enumerate(user["values_priority"]):
        run_query("""
            MERGE (v:Value {name: $value})
            WITH v
            MATCH (u:User {user_id: $user_id})
            MERGE (u)-[r:PRIORITIZES]->(v)
            SET r.priority = $priority, r.week = 0
        """, {
            "value": value,
            "user_id": user["user_id"],
            "priority": idx + 1,
        })

    print(f"온보딩 저장 완료: {user['user_id']}")


def save_weekly_retrospective(user_id: str, week: int, retro: dict):
    """주간 KHT 회고 항목을 Keep/Hard/Try 노드로 Neo4j에 저장"""

    run_query("""
        MATCH (u:User {user_id: $user_id})
        SET u.latest_week = CASE WHEN $week > coalesce(u.latest_week, 0) THEN $week ELSE u.latest_week END
    """, {"user_id": user_id, "week": week})

    for item in retro.get("keep", []):
        run_query("""
            MERGE (k:Keep {text: $text})
            WITH k
            MATCH (u:User {user_id: $user_id})
            MERGE (u)-[r:KEPT]->(k)
            SET r.week = $week
        """, {"text": item, "user_id": user_id, "week": week})

    for item in retro.get("hard", []):
        run_query("""
            MERGE (h:Hard {text: $text})
            WITH h
            MATCH (u:User {user_id: $user_id})
            MERGE (u)-[r:STRUGGLED]->(h)
            SET r.week = $week
        """, {"text": item, "user_id": user_id, "week": week})

    for item in retro.get("try", []):
        run_query("""
            MERGE (t:Try {text: $text})
            WITH t
            MATCH (u:User {user_id: $user_id})
            MERGE (u)-[r:TRIED]->(t)
            SET r.week = $week
        """, {"text": item, "user_id": user_id, "week": week})

    print(f"  KHT 저장 완료: {user_id} - {week}주차")


def save_analysis_result(user_id: str, week: int, analysis: dict):
    """analysis_agent / reframing_agent 결과를 Neo4j에 저장"""

    emotion = analysis.get("emotion", {}).get("primary")
    if emotion:
        run_query("""
            MERGE (e:Emotion {name: $emotion})
            WITH e
            MATCH (u:User {user_id: $user_id})
            MERGE (u)-[r:FEELS]->(e)
            SET r.week = $week,
                r.signal = $signal,
                r.intensity = $intensity
        """, {
            "emotion": emotion,
            "user_id": user_id,
            "week": week,
            "signal": analysis.get("burnout_signal", ""),
            "intensity": analysis.get("emotion", {}).get("intensity", ""),
        })

    for tag in analysis.get("graph_tags", []):
        run_query("""
            MERGE (c:Concern {name: $tag})
            WITH c
            MATCH (u:User {user_id: $user_id})
            MERGE (u)-[r:WORRIES_ABOUT]->(c)
            SET r.week = $week
        """, {"tag": tag, "user_id": user_id, "week": week})

    print(f"  분석 결과 저장 완료: {user_id} - {week}주차")


def save_care_type(user_id: str, care_type: dict):
    """CareType 코드 Neo4j에 저장 (10주 누적 후)"""
    code = care_type.get("code")
    if not code:
        return

    run_query("""
        MERGE (c:CareType {code: $code})
        SET c.type_name = $type_name
        WITH c
        MATCH (u:User {user_id: $user_id})
        MERGE (u)-[r:HAS_TYPE]->(c)
        SET r.week = $week
    """, {
        "code": code,
        "type_name": care_type.get("type_name", ""),
        "user_id": user_id,
        "week": care_type.get("weeks_accumulated", 0),
    })

    print(f"CareType 저장 완료: {user_id} → {code}")


def save_user_report(report: dict):
    """
    B팀 WeeklyReport 스키마 데이터를 Neo4j에 저장.
    routers/graph.py의 POST /api/graph/save 엔드포인트에서 호출.
    """
    user_id = report["user_id"]
    week = report["week"]

    run_query("""
        MERGE (r:WeeklyReport {user_id: $user_id, week: $week})
        SET r.emotion        = $emotion,
            r.emotion_signal = $emotion_signal,
            r.reframing      = $reframing,
            r.value_changed  = $value_changed,
            r.care_type      = $care_type
        WITH r
        MATCH (u:User {user_id: $user_id})
        MERGE (u)-[:SUBMITTED]->(r)
        SET u.latest_week = CASE WHEN $week > coalesce(u.latest_week, 0) THEN $week ELSE u.latest_week END
    """, {
        "user_id": user_id,
        "week": week,
        "emotion": report.get("emotion", ""),
        "emotion_signal": report.get("emotion_signal", ""),
        "reframing": report.get("reframing", ""),
        "value_changed": report.get("value_changed", False),
        "care_type": report.get("care_type"),
    })

    for keyword in report.get("concern_keywords", []):
        run_query("""
            MERGE (c:Concern {name: $keyword})
            WITH c
            MATCH (r:WeeklyReport {user_id: $user_id, week: $week})
            MERGE (r)-[:HAS_CONCERN]->(c)
        """, {"keyword": keyword, "user_id": user_id, "week": week})

    for keyword in report.get("strength_keywords", []):
        run_query("""
            MERGE (s:Strength {name: $keyword})
            WITH s
            MATCH (r:WeeklyReport {user_id: $user_id, week: $week})
            MERGE (r)-[:HAS_STRENGTH]->(s)
        """, {"keyword": keyword, "user_id": user_id, "week": week})

    if report.get("value_changed") and report.get("value_priority"):
        for idx, value in enumerate(report["value_priority"]):
            run_query("""
                MATCH (u:User {user_id: $user_id})
                MERGE (v:Value {name: $value})
                MERGE (u)-[r:PRIORITIZES]->(v)
                SET r.priority = $priority, r.week = $week
            """, {"user_id": user_id, "value": value, "priority": idx + 1, "week": week})

    print(f"  WeeklyReport 저장 완료: {user_id} - {week}주차")


def save_pipeline_result(
    user_id: str,
    week: int,
    keep: list,
    hard: list,
    try_: list,
    result: dict,
) -> None:
    """
    A팀 파이프라인 전체 결과를 Neo4j WeeklyReport 노드로 저장.
    route.py의 POST /api/retrospective 엔드포인트에서 호출.
    """
    analysis = result.get("analysis", {})
    reframing = result.get("reframing", {})
    values = result.get("values", {})
    pattern = result.get("pattern", {})
    care_type = result.get("care_type", {})

    axis_scores = {k: pattern.get(k, {}) for k in ["axis1", "axis2", "axis3", "axis4"]}
    graph_tags = analysis.get("graph_tags", [])

    # WeeklyReport 노드 생성 (keep/hard/try_next 포함하여 get_user_history 쿼리 호환)
    run_query("""
        MERGE (r:WeeklyReport {user_id: $user_id, week: $week})
        SET r.keep                 = $keep,
            r.hard                 = $hard,
            r.try_next             = $try_next,
            r.emotion              = $emotion,
            r.burnout_signal       = $burnout_signal,
            r.recovery_signal      = $recovery_signal,
            r.summary              = $summary,
            r.graph_tags           = $graph_tags,
            r.reframing            = $reframing,
            r.value_change_message = $value_change_message,
            r.axis_scores_json     = $axis_scores_json,
            r.care_type            = $care_type_code
        WITH r
        MATCH (u:User {user_id: $user_id})
        MERGE (u)-[:SUBMITTED]->(r)
        SET u.latest_week = CASE WHEN $week > coalesce(u.latest_week, 0) THEN $week ELSE u.latest_week END
    """, {
        "user_id": user_id,
        "week": week,
        "keep": keep,
        "hard": hard,
        "try_next": try_,
        "emotion": analysis.get("emotion", {}).get("primary", ""),
        "burnout_signal": analysis.get("burnout_signal", ""),
        "recovery_signal": analysis.get("recovery_signal", ""),
        "summary": analysis.get("summary", ""),
        "graph_tags": graph_tags,
        "reframing": reframing.get("overall_message", ""),
        "value_change_message": values.get("update_message"),
        "axis_scores_json": json.dumps(axis_scores, ensure_ascii=False),
        "care_type_code": care_type.get("code") if care_type.get("is_complete") else None,
    })

    # 감정 노드
    emotion_name = analysis.get("emotion", {}).get("primary")
    if emotion_name:
        run_query("""
            MERGE (e:Emotion {name: $emotion})
            WITH e
            MATCH (u:User {user_id: $user_id})
            MERGE (u)-[r:FEELS]->(e)
            SET r.week = $week,
                r.signal = $signal,
                r.intensity = $intensity
        """, {
            "emotion": emotion_name,
            "user_id": user_id,
            "week": week,
            "signal": analysis.get("burnout_signal", ""),
            "intensity": analysis.get("emotion", {}).get("intensity", ""),
        })

    # GraphTag + Concern 노드
    for tag in graph_tags:
        run_query("""
            MERGE (t:GraphTag {name: $tag})
            WITH t
            MATCH (u:User {user_id: $user_id})
            MERGE (u)-[r:HAS_TAG]->(t)
            SET r.week = $week
        """, {"tag": tag, "user_id": user_id, "week": week})

        run_query("""
            MERGE (c:Concern {name: $tag})
            WITH c
            MATCH (u:User {user_id: $user_id})
            MERGE (u)-[r:WORRIES_ABOUT]->(c)
            SET r.week = $week
        """, {"tag": tag, "user_id": user_id, "week": week})

    # 강점 키워드 (reframing 결과)
    for item in reframing.get("reframing", []):
        for keyword in item.get("strength_keywords", []):
            run_query("""
                MERGE (s:Strength {name: $keyword})
                WITH s
                MATCH (u:User {user_id: $user_id})
                MERGE (u)-[r:HAS_STRENGTH]->(s)
                SET r.week = $week
            """, {"keyword": keyword, "user_id": user_id, "week": week})

    print(f"  파이프라인 결과 저장 완료: {user_id} - {week}주차")


def load_dummy_data():
    """더미 데이터 전체를 Neo4j에 저장 (온보딩 + KHT 항목)"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dummy_path = os.path.join(base_dir, "routers", "dummy_data.json")

    with open(dummy_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"총 {len(data)}명 저장 시작...\n")

    for user in data:
        save_user_onboarding(user)

        for retro in user.get("retrospectives", []):
            save_weekly_retrospective(
                user_id=user["user_id"],
                week=retro["week"],
                retro=retro,
            )

    print("\n더미 데이터 저장 완료!")


if __name__ == "__main__":
    load_dummy_data()
