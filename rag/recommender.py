from typing import Any

from rag.retriever import build_graph_rag_context, find_similar_users


def recommend_users(user_id: str, top_k: int = 3) -> list[dict[str, Any]]:
    return find_similar_users(user_id=user_id, top_k=top_k)


def get_agent_context(user_id: str, current_week: dict[str, Any] | None = None) -> str:
    return build_graph_rag_context(user_id=user_id, current_week=current_week)
