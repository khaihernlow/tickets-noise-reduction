import json
import os
import requests
from typing import Optional


BASE_URL = "https://ai.hatz.ai/v1"


class HatzAIError(Exception):
    pass


class HatzAIClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ["HATZAI_API_KEY"]
        self.model = model or os.environ.get("HATZAI_MODEL", "claude-sonnet-4-6")
        self._session = requests.Session()
        self._session.headers.update({"X-API-Key": self.api_key, "Content-Type": "application/json"})

    def list_models(self) -> list[dict]:
        resp = self._session.get(f"{BASE_URL}/chat/models")
        resp.raise_for_status()
        data = resp.json()
        # handle both {"data": [...]} and plain list responses
        return data.get("data", data) if isinstance(data, dict) else data

    def chat(
        self,
        user_message: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = self._session.post(f"{BASE_URL}/chat/completions", json=payload)

        if not resp.ok:
            raise HatzAIError(f"HatzAI API error {resp.status_code}: {resp.text}")

        data = resp.json()

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise HatzAIError(f"Unexpected response shape: {data}") from e

    def chat_json(
        self,
        user_message: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> dict:
        """Chat and parse the response as JSON. Retries once if parsing fails."""
        raw = self.chat(user_message, system=system, temperature=temperature, max_tokens=max_tokens)
        raw = raw.strip()

        # strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # ask the model to fix its own output
            fix_prompt = f"Return ONLY valid JSON with no explanation or markdown:\n{raw}"
            fixed = self.chat(fix_prompt, temperature=0.0, max_tokens=max_tokens)
            fixed = fixed.strip()
            if fixed.startswith("```"):
                lines = fixed.splitlines()
                fixed = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return json.loads(fixed)
