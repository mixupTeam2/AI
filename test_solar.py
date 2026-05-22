import requests

API_KEY = "your-api-key"

response = requests.post(
    "https://api.upstage.ai/v1/solar/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": "solar-pro2-preview",  # or solar-pro
        "messages": [{"role": "user", "content": "안녕"}]
    }
)
print(response.json())