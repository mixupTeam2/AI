import requests
import base64
import os
from dotenv import load_dotenv

load_dotenv()

URL = os.getenv("NEO4J_QUERY_URL")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def get_headers():
    credentials = base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
    return {
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/json"
    }

def run_query(statement: str, parameters: dict = {}):
    body = {
        "statement": statement,
        "parameters": parameters
    }
    response = requests.post(URL, json=body, headers=get_headers())
    if response.status_code in [200, 202]:
        return response.json()
    else:
        raise Exception(f"쿼리 실패: {response.status_code} {response.text}")

def create_schema():
    queries = [
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
        "CREATE CONSTRAINT caretype_code IF NOT EXISTS FOR (c:CareType) REQUIRE c.code IS UNIQUE",
        "CREATE CONSTRAINT emotion_name IF NOT EXISTS FOR (e:Emotion) REQUIRE e.name IS UNIQUE",
        "CREATE CONSTRAINT value_name IF NOT EXISTS FOR (v:Value) REQUIRE v.name IS UNIQUE",
        "CREATE CONSTRAINT concern_name IF NOT EXISTS FOR (c:Concern) REQUIRE c.name IS UNIQUE",
    ]
    for q in queries:
        try:
            run_query(q)
            print(f"완료: {q[:50]}...")
        except Exception as e:
            print(f"스킵 (이미 존재): {e}")
    print("\n스키마 생성 완료!")

def check_data():
    result = run_query("MATCH (u:User) RETURN u.user_id, u.latest_week")
    print("\n저장된 유저 목록:")
    for row in result["data"]["values"]:
        print(f"  - {row[0]} / {row[1]}주차")

    result2 = run_query("MATCH ()-[r]->() RETURN count(r) AS total")
    print(f"\n전체 관계 수: {result2['data']['values'][0][0]}개")

if __name__ == "__main__":
    create_schema()
    check_data()