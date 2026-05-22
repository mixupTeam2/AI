from graph.schema import run_query

def find_similar_users(user_id: str, top_k: int = 3) -> list:
    """
    유사 유저 탐색 (가치관 + 감정 + CareType 기반 유사도 계산)
    """
    result = run_query("""
        MATCH (target:User {user_id: $user_id})
        MATCH (other:User)
        WHERE other.user_id <> $user_id

        // 가치관 유사도 (같은 Value 노드 공유)
        OPTIONAL MATCH (target)-[tv:PRIORITIZES]->(v:Value)<-[ov:PRIORITIZES]-(other)
        WITH target, other,
             count(v) AS shared_values,
             sum(abs(tv.priority - ov.priority)) AS priority_diff

        // 감정 유사도 (같은 Emotion 노드 공유)
        OPTIONAL MATCH (target)-[:FEELS]->(e:Emotion)<-[:FEELS]-(other)
        WITH target, other, shared_values, priority_diff,
             count(e) AS shared_emotions

        // CareType 유사도
        OPTIONAL MATCH (target)-[:HAS_TYPE]->(c:CareType)<-[:HAS_TYPE]-(other)
        WITH other, shared_values, priority_diff, shared_emotions,
             count(c) AS shared_type

        // 유사도 점수 계산
        WITH other,
             (shared_values * 3 + shared_emotions * 2 + shared_type * 2
              - coalesce(priority_diff, 0) * 0.5) AS similarity_score

        ORDER BY similarity_score DESC
        LIMIT $top_k

        // 추천 유저 정보 수집
        MATCH (other)-[:FEELS]->(e:Emotion)
        MATCH (other)-[:PRIORITIZES]->(v:Value)
        OPTIONAL MATCH (other)-[:HAS_TYPE]->(ct:CareType)
        OPTIONAL MATCH (other)-[:WORRIES_ABOUT]->(con:Concern)

        RETURN other.user_id AS user_id,
               ct.code AS care_type,
               e.name AS emotion,
               collect(DISTINCT v.name) AS value_priority,
               collect(DISTINCT con.name) AS concern_keywords,
               similarity_score
    """, {"user_id": user_id, "top_k": top_k})

    recommendations = []
    for row in result["data"]["values"]:
        recommendations.append({
            "user_id": row[0],
            "care_type": row[1],
            "emotion": row[2],
            "value_priority": row[3],
            "concern_keywords": row[4],
            "similarity_score": round(row[5], 2)
        })

    return recommendations


if __name__ == "__main__":
    result = find_similar_users("user_001")
    print("\nuser_001 추천 유저:")
    for r in result:
        print(f"  - {r['user_id']} | 유사도: {r['similarity_score']} | 감정: {r['emotion']} | 가치관: {r['value_priority']}")