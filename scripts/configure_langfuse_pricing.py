"""Create the project's dated DeepSeek cost estimate in local Langfuse."""

import os
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
values = {**dotenv_values(ROOT / ".env"), **os.environ}
base_url = str(values.get("LANGFUSE_BASE_URL") or "http://localhost:3000").rstrip("/")

if base_url.startswith("http://host.docker.internal"):
    base_url = base_url.replace("http://host.docker.internal", "http://localhost", 1)
public_key = values.get("LANGFUSE_PUBLIC_KEY")
secret_key = values.get("LANGFUSE_SECRET_KEY")
if not public_key or not secret_key:
    raise SystemExit("缺少 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY")

auth = (str(public_key), str(secret_key))
endpoint = f"{base_url}/api/public/models"
model_name = "deepseek-chat-peak-estimate-2026-09-17"
response = requests.get(endpoint, auth=auth, timeout=15)
response.raise_for_status()
if any(item.get("modelName") == model_name for item in response.json().get("data", [])):
    print("DeepSeek 费用规则已存在。")
    raise SystemExit(0)


response = requests.post(
    endpoint,
    auth=auth,
    timeout=15,
    json={
        "modelName": model_name,
        "matchPattern": "(?i)^deepseek-chat$",
        "unit": "TOKENS",
        "inputPrice": 0.30 / 1_000_000,
        "outputPrice": 1.20 / 1_000_000,
    },
)
response.raise_for_status()
print("已创建 DeepSeek 峰值费用估算规则。")
