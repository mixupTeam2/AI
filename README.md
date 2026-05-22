# CareType AI

FastAPI 기반 주간 KHT 회고 분석 API입니다. Solar 멀티 에이전트와 Neo4j Graph DB를 연결해 Graph RAG 문맥을 주입합니다.

## 실행

1. `.env.example`을 참고해 `.env`에 Solar와 Neo4j 값을 넣습니다.
2. 의존성을 설치합니다.

```bash
pip install -r requirements.txt
```

3. Neo4j 제약조건을 생성합니다.

```bash
python -m graph.schema
```

4. 서버를 실행합니다.

```bash
uvicorn main:app --reload
```

## RAG 흐름

`POST /api/onboarding`은 유저의 스펙과 가치관 우선순위를 Graph DB에 저장합니다.

`POST /api/retrospective`는 다음 순서로 동작합니다.

1. Graph DB에서 현재 유저의 과거 회고, 가치관, 유사 유저, 누적 CareType 축 점수를 조회합니다.
2. 조회 결과를 `rag_context`로 만들어 에이전트 오케스트레이터에 주입합니다.
3. Solar 오케스트레이터가 실행할 에이전트 순서를 정합니다.
4. 분석, 리프레이밍, 패턴, 가치관 에이전트가 실행됩니다.
5. 패턴 에이전트 결과는 CareType 4축 점수로 저장되고, 10주 이상 누적되면 `type_agent`가 최종 CareType을 생성합니다.
6. 회고 입력, 에이전트 결과, `ui_summary`, graph tag, 감정, 가치관, 관심사 관계를 Graph DB에 다시 저장합니다.

프론트는 우선 `result.ui_summary`만 사용해도 됩니다. 전체 원본 결과는 `result.analysis`, `result.reframing`, `result.pattern`, `result.values`, `result.supervisor`에 남아 있습니다.

유사 유저 추천은 `GET /api/recommend/{user_id}` 또는 `GET /api/rag/{user_id}`로 확인할 수 있습니다.
