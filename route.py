import json
import os
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from graph.pipeline import save_pipeline_result
from graph.queries import get_current_value_priority
from graph.schema import run_query
from orchestrator import run_pipeline
from rag.retriever import find_similar_users

router = APIRouter()

SOLAR_URL = "https://api.upstage.ai/v1/solar/chat/completions"
API_KEY = os.getenv("SOLAR_API_KEY") or os.getenv("UPSTAGE_API_KEY", "")


class Spec(BaseModel):
    gpa: Optional[float] = None
    activities: Optional[list[str]] = []
    lab_experience: Optional[bool] = False
    certifications: Optional[list[str]] = []


VALUES_OPTIONS = [
    "근무지",
    "워라밸",
    "도메인",
    "직무",
    "연봉",
    "기업규모",
    "안정성",
]


class OnboardingRequest(BaseModel):
    user_id: str
    spec: Spec
    values_priority: list[str] = Field(
        ...,
        description="7개 가치관 항목을 우선순위 순서대로 정렬한 리스트",
        examples=[["워라밸", "직무", "도메인", "근무지", "안정성", "연봉", "기업규모"]],
    )


class RetrospectiveRequest(BaseModel):
    user_id: str
    week: int = Field(..., ge=1, description="회고 주차 (1부터 시작)")
    keep: list[str] = Field(..., description="잘 된 것")
    hard: list[str] = Field(..., description="힘들었던 것")
    try_next: list[str] = Field(..., alias="try", description="다음 주 시도할 것")
    values_priority: Optional[list[str]] = None

    class Config:
        populate_by_name = True


class ValuesUpdateRequest(BaseModel):
    values_priority: list[str] = Field(..., description="업데이트된 가치관 우선순위 리스트")


def _dump_model(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _priority_from_list(values: list[str]) -> dict[str, int]:
    return {value: idx + 1 for idx, value in enumerate(values)}


def _default_priority() -> dict[str, int]:
    return _priority_from_list(VALUES_OPTIONS)


def _load_current_priority(user_id: str) -> dict[str, int]:
    try:
        priority = get_current_value_priority(user_id)
        return priority or _default_priority()
    except Exception:
        return _default_priority()


def _save_onboarding_to_graph(req: OnboardingRequest) -> None:
    spec = _dump_model(req.spec)
    run_query(
        """
        MERGE (u:User {user_id: $user_id})
        SET u.spec_json = $spec_json
        """,
        {"user_id": req.user_id, "spec_json": json.dumps(spec, ensure_ascii=False)},
    )
    for idx, value in enumerate(req.values_priority):
        run_query(
            """
            MATCH (u:User {user_id: $user_id})
            MERGE (v:Value {name: $value})
            MERGE (u)-[r:PRIORITIZES]->(v)
            SET r.priority = $priority, r.week = 0
            """,
            {"user_id": req.user_id, "value": value, "priority": idx + 1},
        )


def _solar_chat(system_prompt: str, user_content: str, model: str = "solar-pro3") -> str:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="SOLAR_API_KEY or UPSTAGE_API_KEY not set")
    response = requests.post(
        SOLAR_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()["choices"][0]["message"]["content"]


@router.post("/onboarding", summary="스펙 + 가치관 우선순위 온보딩")
def onboarding(req: OnboardingRequest):
    invalid = [value for value in req.values_priority if value not in VALUES_OPTIONS]
    if invalid:
        raise HTTPException(status_code=422, detail=f"유효하지 않은 가치관 항목: {invalid}")
    if len(req.values_priority) != 7:
        raise HTTPException(status_code=422, detail="가치관 항목은 7개여야 합니다")

    graph_saved = True
    graph_error = None
    try:
        _save_onboarding_to_graph(req)
    except Exception as exc:
        graph_saved = False
        graph_error = str(exc)

    payload = {
        "user_id": req.user_id,
        "message": "온보딩 완료",
        "spec": _dump_model(req.spec),
        "values_priority": req.values_priority,
        "graph_saved": graph_saved,
    }
    if graph_error:
        payload["graph_error"] = graph_error
    return payload


@router.post("/retrospective", summary="주간 KHT 회고 제출 및 멀티 에이전트 분석")
def submit_retrospective(req: RetrospectiveRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="SOLAR_API_KEY or UPSTAGE_API_KEY not set")

    current_priority = (
        _priority_from_list(req.values_priority)
        if req.values_priority
        else _load_current_priority(req.user_id)
    )

    try:
        result = run_pipeline(
            keep=req.keep,
            hard=req.hard,
            try_=req.try_next,
            current_priority=current_priority,
            api_key=API_KEY,
            user_id=req.user_id,
            week=req.week,
        )
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 502
        raise HTTPException(status_code=status_code, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # 파이프라인 결과를 Neo4j에 저장 (실패해도 응답은 정상 반환)
    graph_saved = True
    graph_error = None
    try:
        save_pipeline_result(
            user_id=req.user_id,
            week=req.week,
            keep=req.keep,
            hard=req.hard,
            try_=req.try_next,
            result=result,
        )
    except Exception as exc:
        graph_saved = False
        graph_error = str(exc)
        print(f"[Neo4j] 저장 실패: {exc}")

    response_payload = {
        "user_id": req.user_id,
        "week": req.week,
        "result": result,
        "graph_saved": graph_saved,
    }
    if graph_error:
        response_payload["graph_error"] = graph_error
    return response_payload


@router.get("/recommend/{user_id}", summary="유사 유저 3명 추천 (Graph DB RAG)")
def get_recommendations(user_id: str):
    try:
        return {
            "user_id": user_id,
            "recommended_users": find_similar_users(user_id, top_k=3),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
