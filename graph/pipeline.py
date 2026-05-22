import json
import os
from graph.schema import run_query

def save_user_report(report: dict):
    """분석 결과 1건을 Neo4j에 저장"""

    # 1. User 노드 생성
    run_query("""
        MERGE (u:User {user_id: $user_id})
        SET u.latest_week = $week
    """, {"user_id": report["user_id"], "week": report["week"]})

    # 2. CareType 노드 생성 및 연결
    if report.get("care_type"):
        run_query("""
            MERGE (c:CareType {code: $code})
            WITH c
            MATCH (u:User {user_id: $user_id})
            MERGE (u)-[r:HAS_TYPE]->(c)
            SET r.week = $week
        """, {
            "code": report["care_type"],
            "user_id": report["user_id"],
            "week": report["week"]
        })

    # 3. Emotion 노드 생성 및 연결
    run_query("""
        MERGE (e:Emotion {name: $emotion})
        WITH e
        MATCH (u:User {user_id: $user_id})
        MERGE (u)-[r:FEELS]->(e)
        SET r.week = $week, r.signal = $signal
    """, {
        "emotion": report["emotion"],
        "user_id": report["user_id"],
        "week": report["week"],
        "signal": report["emotion_signal"]
    })

    # 4. Value 노드 생성 및 연결 (우선순위 포함)
    for idx, value in enumerate(report["value_priority"]):
        run_query("""
            MERGE (v:Value {name: $value})
            WITH v
            MATCH (u:User {user_id: $user_id})
            MERGE (u)-[r:PRIORITIZES]->(v)
            SET r.priority = $priority, r.week = $week
        """, {
            "value": value,
            "user_id": report["user_id"],
            "priority": idx + 1,
            "week": report["week"]
        })

    # 5. Concern 노드 생성 및 연결
    for concern in report["concern_keywords"]:
        run_query("""
            MERGE (c:Concern {name: $concern})
            WITH c
            MATCH (u:User {user_id: $user_id})
            MERGE (u)-[r:WORRIES_ABOUT]->(c)
            SET r.week = $week
        """, {
            "concern": concern,
            "user_id": report["user_id"],
            "week": report["week"]
        })

    print(f"저장 완료: {report['user_id']} - {report['week']}주차")

def load_dummy_data():
    """더미 데이터 전체 Neo4j에 저장"""
    with open("dummy_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"총 {len(data)}건 저장 시작...")
    for report in data:
        save_user_report(report)
    print("\n더미 데이터 저장 완료!")

if __name__ == "__main__":
    load_dummy_data()