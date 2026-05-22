import base64
import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

URL = os.getenv("NEO4J_QUERY_URL")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")
TIMEOUT_SECONDS = float(os.getenv("NEO4J_TIMEOUT_SECONDS", "20"))


class GraphConfigError(RuntimeError):
    """Raised when Neo4j connection settings are missing."""


def _require_config() -> None:
    missing = [
        name
        for name, value in {
            "NEO4J_QUERY_URL": URL,
            "NEO4J_USER": USER,
            "NEO4J_PASSWORD": PASSWORD,
        }.items()
        if not value
    ]
    if missing:
        raise GraphConfigError(f"Missing Neo4j env vars: {', '.join(missing)}")


def get_headers() -> dict[str, str]:
    _require_config()
    credentials = base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
    return {
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/json",
    }


def run_query(statement: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one Cypher statement through the Neo4j Query API."""
    response = requests.post(
        URL,
        json={"statement": statement, "parameters": parameters or {}},
        headers=get_headers(),
        timeout=TIMEOUT_SECONDS,
    )
    if response.status_code in [200, 202]:
        return response.json()
    raise RuntimeError(f"Neo4j query failed: {response.status_code} {response.text}")


def result_values(result: dict[str, Any]) -> list[list[Any]]:
    return result.get("data", {}).get("values", [])


def create_schema() -> None:
    queries = [
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
        "CREATE CONSTRAINT caretype_code IF NOT EXISTS FOR (c:CareType) REQUIRE c.code IS UNIQUE",
        "CREATE CONSTRAINT emotion_name IF NOT EXISTS FOR (e:Emotion) REQUIRE e.name IS UNIQUE",
        "CREATE CONSTRAINT value_name IF NOT EXISTS FOR (v:Value) REQUIRE v.name IS UNIQUE",
        "CREATE CONSTRAINT concern_name IF NOT EXISTS FOR (c:Concern) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT report_key IF NOT EXISTS FOR (r:WeeklyReport) REQUIRE (r.user_id, r.week) IS UNIQUE",
        "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:GraphTag) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT strength_name IF NOT EXISTS FOR (s:Strength) REQUIRE s.name IS UNIQUE",
    ]
    for query in queries:
        try:
            run_query(query)
            print(f"created: {query[:70]}...")
        except Exception as exc:
            print(f"skipped: {exc}")
    print("\nNeo4j schema is ready.")


def check_data() -> None:
    result = run_query("MATCH (u:User) RETURN u.user_id, u.latest_week ORDER BY u.user_id")
    print("\nUsers:")
    for row in result_values(result):
        print(f"  - {row[0]} / week {row[1]}")

    result2 = run_query("MATCH ()-[r]->() RETURN count(r) AS total")
    total = result_values(result2)[0][0] if result_values(result2) else 0
    print(f"\nRelationships: {total}")


if __name__ == "__main__":
    create_schema()
    check_data()
