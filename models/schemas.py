from pydantic import BaseModel
from typing import List, Optional

class WeeklyReport(BaseModel):
    """A담당 분석 결과 수신 포맷"""
    user_id: str
    week: int
    care_type: Optional[str] = None
    emotion: str
    emotion_signal: str
    concern_keywords: List[str]
    value_priority: List[str]
    value_changed: bool
    reframing: str
    strength_keywords: List[str]

class UpdateValueRequest(BaseModel):
    """가치관 업데이트 요청 포맷"""
    user_id: str
    new_value_priority: List[str]
    week: int

class RecommendedUser(BaseModel):
    """추천 유저 1명 데이터"""
    user_id: str
    care_type: Optional[str]
    emotion: str
    concern_keywords: List[str]
    value_priority: List[str]
    similarity_score: float

class RecommendResponse(BaseModel):
    """추천 응답 포맷"""
    recommendations: List[RecommendedUser]