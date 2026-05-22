import os
import json
import requests
from neo4j import GraphDatabase
from typing import Optional

UPSTAGE_EMBED_URL = "https://api.upstage.ai/v1/embeddings"
EMBED_MODEL = "solar-embedding-1-large"
EMBED_DIMENSIONS = 4096


class GraphRAG:
    """
    Neo4j + Upstage Embedding 기반 유사 유저 RAG 시스템.
    neo4j driver + requests 직접 사용.

    저장 구조:
      (:User {user_id})
        -[:HAS_WEEK]->
      (:Week {user_id, week, burnout_signal, summary, embedding})
        -[:HAS_KEYWORD]->
      (:Keyword {name, type})
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        upstage_api_key: Optional[str] = None,
    ):
        self.driver = GraphDatabase.driver(
            neo4j_uri or os.environ["NEO4J_URI"],
            auth=(
                neo4j_user or os.environ["NEO4J_USER"],
                neo4j_password or os.environ["NEO4J_PASSWORD"],
            ),
        )
        self.api_key = upstage_api_key or os.environ["UPSTAGE_API_KEY"]
        self._ensure_vector_index()

    # ── 임베딩 ─────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        response = requests.post(
            UPSTAGE_EMBED_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": EMBED_MODEL, "input": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    # ── 인덱스 ─────────────────────────────────────────────────────────

    def _ensure_vector_index(self):
        with self.driver.session() as session:
            session.run(f"""
                CREATE VECTOR INDEX week_embeddings IF NOT EXISTS
                FOR (w:Week) ON (w.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {EMBED_DIMENSIONS},
                    `vector.similarity_function`: 'cosine'
                }}}}
            """)

    # ── 텍스트 요약 생성 ───────────────────────────────────────────────

    def _build_summary(self, agent_results: dict) -> str:
        parts = []

        analysis = agent_results.get("analysis", {})
        if analysis:
            emotion = analysis.get("emotion", {})
            parts.append(f"감정: {emotion.get('primary', '')} {' '.join(emotion.get('secondary', []))}")
            parts.append(f"번아웃신호: {analysis.get('burnout_signal', '')}")
            parts.append(f"고민도메인: {analysis.get('context', {}).get('domain', '')}")

        reframing = agent_results.get("reframing", {})
        if reframing:
            keywords = []
            for item in reframing.get("reframing", []):
                keywords.extend(item.get("strength_keywords", []))
            if keywords:
                parts.append(f"강점키워드: {' '.join(keywords)}")

        pattern_result = agent_results.get("pattern", {})
        if pattern_result:
            scores = [
                pattern_result.get("axis1", {}).get("score", 0),
                pattern_result.get("axis2", {}).get("score", 0),
                pattern_result.get("axis3", {}).get("score", 0),
                pattern_result.get("axis4", {}).get("score", 0),
            ]
            parts.append(f"축점수: {' '.join(f'축{i+1}:{s:+d}' for i, s in enumerate(scores))}")

        values = agent_results.get("values", {})
        if values and values.get("change_detected"):
            parts.append(f"가치관변화: {' '.join(values.get('changed_values', []))}")

        return ". ".join(parts)

    # ── 저장 ──────────────────────────────────────────────────────────

    def save_weekly_result(self, user_id: str, week: int, agent_results: dict):
        """에이전트 결과를 Neo4j에 저장합니다."""
        summary = self._build_summary(agent_results)
        embedding = self._embed(summary)

        analysis = agent_results.get("analysis", {})
        reframing = agent_results.get("reframing", {})
        pattern = agent_results.get("pattern", {})

        strength_keywords = []
        emotion_tags = []
        for item in reframing.get("reframing", []):
            strength_keywords.extend(item.get("strength_keywords", []))
            emotion_tags.extend(item.get("emotion_tags", []))

        with self.driver.session() as session:
            session.run("""
                MERGE (u:User {user_id: $user_id})
                MERGE (w:Week {user_id: $user_id, week: $week})
                SET w.burnout_signal = $burnout_signal,
                    w.summary = $summary,
                    w.embedding = $embedding
                MERGE (u)-[:HAS_WEEK]->(w)
            """, {
                "user_id": user_id,
                "week": week,
                "burnout_signal": analysis.get("burnout_signal", ""),
                "summary": summary,
                "embedding": embedding,
            })

            for kw in strength_keywords:
                session.run("""
                    MERGE (k:Keyword {name: $name, type: 'strength'})
                    WITH k
                    MATCH (w:Week {user_id: $user_id, week: $week})
                    MERGE (w)-[:HAS_KEYWORD]->(k)
                """, {"name": kw, "user_id": user_id, "week": week})

            for tag in emotion_tags:
                session.run("""
                    MERGE (k:Keyword {name: $name, type: 'emotion'})
                    WITH k
                    MATCH (w:Week {user_id: $user_id, week: $week})
                    MERGE (w)-[:HAS_KEYWORD]->(k)
                """, {"name": tag, "user_id": user_id, "week": week})

    # ── 조회 ──────────────────────────────────────────────────────────

    def get_similar_context(
        self,
        current_agent_results: dict,
        current_user_id: str,
        k: int = 3,
    ) -> str:
        """오케스트레이터 rag_context용 — 유사 유저 패턴 문자열 반환."""
        users = self.get_similar_users(current_agent_results, current_user_id, k)
        if not users:
            return ""
        lines = [
            f"유사유저{i+1} (week {u['week']}): {u['summary']}"
            for i, u in enumerate(users)
        ]
        return "유사 유저 패턴:\n" + "\n".join(lines)

    def get_similar_users(
        self,
        current_agent_results: dict,
        current_user_id: str,
        k: int = 3,
    ) -> list[dict]:
        """유저 추천 에이전트용 — 유사 유저 리스트 반환."""
        summary = self._build_summary(current_agent_results)
        embedding = self._embed(summary)

        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('week_embeddings', $top_k, $embedding)
                YIELD node, score
                WHERE node.user_id <> $current_user_id
                RETURN node.user_id AS user_id,
                       node.week AS week,
                       node.summary AS summary,
                       score
                LIMIT $k
            """, {
                "embedding": embedding,
                "current_user_id": current_user_id,
                "top_k": k + 5,
                "k": k,
            })

            return [
                {
                    "user_id": r["user_id"],
                    "week": r["week"],
                    "summary": r["summary"],
                    "similarity_score": round(r["score"], 4),
                }
                for r in result
            ]

    def close(self):
        self.driver.close()
