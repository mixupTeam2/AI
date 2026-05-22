from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
from route import router

load_dotenv()

app = FastAPI(title="CareType API")
app.include_router(router, prefix="/api")

API_KEY = os.getenv("SOLAR_API_KEY", "")
SOLAR_URL = "https://api.upstage.ai/v1/solar/chat/completions"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(message: str = Form(...), model: str = Form("solar-pro3")):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="SOLAR_API_KEY not set")

    response = requests.post(
        SOLAR_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": message}],
        }
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()

# 서버 실행 명령어
# uvicorn main:app --reload