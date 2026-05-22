from fastapi import APIRouter, HTTPException
from rag.retriever import find_similar_users

router = APIRouter()

@router.get("/{user_id}")
async def recommend_users(user_id: str):
    try:
        result = find_similar_users(user_id, top_k=3)
        if not result:
            return {"recommendations": []}
        return {"recommendations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/type/{user_id}")
async def get_user_type(user_id: str):
    from graph.schema import run_query
    try:
        result = run_query("""
            MATCH (u:User {user_id: $user_id})
            OPTIONAL MATCH (u)-[:HAS_TYPE]->(c:CareType)
            OPTIONAL MATCH (u)-[:PRIORITIZES]->(v:Value)
            RETURN u.user_id, c.code, collect(v.name) AS value_priority, u.latest_week
        """, {"user_id": user_id})

        if not result["data"]["values"]:
            raise HTTPException(status_code=404, detail="유저를 찾을 수 없어요")

        row = result["data"]["values"][0]
        return {
            "user_id": row[0],
            "care_type": row[1],
            "value_priority": row[2],
            "latest_week": row[3]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))