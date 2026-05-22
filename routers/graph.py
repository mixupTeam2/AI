from fastapi import APIRouter, HTTPException
from models.schemas import WeeklyReport, UpdateValueRequest
from graph.pipeline import save_user_report
from graph.schema import run_query

router = APIRouter()

@router.post("/save")
async def save_report(report: WeeklyReport):
    try:
        save_user_report(report.dict())
        return {"status": "ok", "user_id": report.user_id, "week": report.week}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update-value")
async def update_value(request: UpdateValueRequest):
    try:
        for idx, value in enumerate(request.new_value_priority):
            run_query("""
                MATCH (u:User {user_id: $user_id})
                MERGE (v:Value {name: $value})
                MERGE (u)-[r:PRIORITIZES]->(v)
                SET r.priority = $priority, r.week = $week
            """, {
                "user_id": request.user_id,
                "value": value,
                "priority": idx + 1,
                "week": request.week
            })
        return {"status": "ok", "user_id": request.user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))