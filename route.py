from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import requests
import os

router = APIRouter()

SOLAR_URL = "https://api.upstage.ai/v1/solar/chat/completions"
API_KEY = os.getenv("SOLAR_API_KEY", "")

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class Spec(BaseModel):
    gpa: Optional[float] = None
    activities: Optional[list[str]] = []
    lab_experience: Optional[bool] = False
    certifications: Optional[list[str]] = []

VALUES_OPTIONS = [
    "근무지", "워라밸", "도메인", "직무", "연봉", "기업규모", "안정성"
]

class OnboardingRequest(BaseModel):
    user_id: str
    spec: Spec
    values_priority: list[str] = Field(
        ...,
        description="7개 가치관 항목을 우선순위 순서대로 정렬한 리스트",
        example=["근무지", "워라밸", "도메인", "직무", "연봉", "기업규모", "안정성"],
    )

class RetrospectiveRequest(BaseModel):
    user_id: str
    week: int = Field(..., ge=1, description="회고 주차 (1부터 시작)")
    keep: list[str] = Field(..., description="잘 된 것")
    hard: list[str] = Field(..., description="힘들었던 것")
    try_next: list[str] = Field(..., alias="try", description="다음 주 시도할 것")

    class Config:
        populate_by_name = True

class ValuesUpdateRequest(BaseModel):
    values_priority: list[str] = Field(
        ..., description="업데이트된 가치관 우선순위 리스트"
    )

# ---------------------------------------------------------------------------
# Helper: Solar Pro3 single call
# ---------------------------------------------------------------------------

def _solar_chat(system_prompt: str, user_content: str, model: str = "solar-pro3") -> str:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="SOLAR_API_KEY not set")
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

# ---------------------------------------------------------------------------
# Routes: Onboarding
# ---------------------------------------------------------------------------

@router.post("/onboarding", summary="스펙 + 가치관 우선순위 온보딩")
def onboarding(req: OnboardingRequest):
    invalid = [v for v in req.values_priority if v not in VALUES_OPTIONS]
    if invalid:
        raise HTTPException(status_code=422, detail=f"유효하지 않은 가치관 항목: {invalid}")
    if len(req.values_priority) != 7:
        raise HTTPException(status_code=422, detail="가치관 항목은 7개여야 합니다")

    # TODO: 실제 DB 저장 (Supabase)
    return {
        "user_id": req.user_id,
        "message": "온보딩 완료",
        "spec": req.spec.model_dump(),
        "values_priority": req.values_priority,
    }

# ---------------------------------------------------------------------------
# Routes: Weekly Retrospective (KHT) + Multi-Agent Analysis
# ---------------------------------------------------------------------------

@router.post("/retrospective", summary="주간 KHT 회고 제출 및 멀티 에이전트 분석")
def submit_retrospective(req: RetrospectiveRequest):
    kht_text = (
        f"[Keep]\n" + "\n".join(f"- {k}" for k in req.keep) + "\n\n"
        f"[Hard]\n" + "\n".join(f"- {h}" for h in req.hard) + "\n\n"
        f"[Try]\n" + "\n".join(f"- {t}" for t in req.try_next)
    )

    # --- Agent 1: 분석 에이전트 ---
    # from agent.analysis import run as analysis_run
    analysis = _solar_chat(
        system_prompt=(
            "당신은 취준생의 KHT 회고를 분석하는 전문 상담 AI입니다. "
            "회고 텍스트에서 감정 상태(긍정/부정/중립)와 번아웃 신호(높음/보통/낮음)를 판단하세요. "
            "JSON 형식으로 응답하세요: {\"emotion\": \"...\", \"burnout_risk\": \"...\", \"summary\": \"...\"}"
        ),
        user_content=kht_text,
    )

    # --- Agent 2: 리프레이밍 에이전트 ---
    # from agent.reframing import run as reframing_run
    reframing = _solar_chat(
        system_prompt=(
            "당신은 취준생의 부정적 경험을 강점 언어로 리프레이밍하는 코치입니다. "
            "Hard 항목을 성장 가능성의 언어로 전환하고, 강점 키워드 2~3개를 추출하세요. "
            "JSON 형식으로 응답하세요: {\"reframed_message\": \"...\", \"strength_keywords\": [...]}"
        ),
        user_content=f"[Hard 항목]\n" + "\n".join(f"- {h}" for h in req.hard),
    )

    # --- Agent 3: 가치관 추적 에이전트 ---
    # from agent.values_tracker import run as values_tracker_run
    values_change = _solar_chat(
        system_prompt=(
            "당신은 취준생의 회고에서 가치관 변화를 감지하는 AI입니다. "
            "텍스트에서 가치관(근무지/워라밸/도메인/직무/연봉/기업규모/안정성) 관련 변화 신호를 찾으세요. "
            "JSON 형식으로 응답하세요: {\"detected_change\": true/false, \"value_item\": \"...\", \"suggestion\": \"...\"}"
        ),
        user_content=kht_text,
    )

    # --- Agent 4: 패턴 추적 에이전트 (누적 필요 → 현재는 단주차 분석) ---
    # from agent.pattern_tracker import run as pattern_tracker_run
    pattern = _solar_chat(
        system_prompt=(
            "당신은 취준생의 주간 회고 패턴을 분석하는 AI입니다. "
            "이번 주 회고에서 에너지 회복 신호 또는 반복 패턴을 간단히 분석하세요. "
            "JSON 형식으로 응답하세요: {\"energy_trend\": \"...\", \"pattern_note\": \"...\"}"
        ),
        user_content=kht_text,
    )

    # TODO: week >= 10 시 유형생성 에이전트 호출 → CareType 코드 생성
    # from agent.type_generator import run as type_generator_run

    # TODO: DB에 회고 저장 + 분석 결과 저장

    return {
        "user_id": req.user_id,
        "week": req.week,
        "analysis": analysis,
        "reframing": reframing,
        "values_change_detection": values_change,
        "pattern": pattern,
    }


@router.get("/recommend/{user_id}", summary="유사 유저 3명 추천 (Graph DB RAG)")
def get_recommendations(user_id: str):
    # TODO: Neo4j Graph DB RAG 기반 유사 유저 검색
    # from agent.recommender import run as recommender_run
    return {
        "user_id": user_id,
        "recommended_users": [],  # placeholder — Graph DB 연결 후 채워질 예정
        "message": "추천 유저 조회 (Graph DB 연동 전 placeholder)",
    }
