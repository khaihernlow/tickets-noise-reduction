import json
import os
import time
import requests
from typing import Optional


BASE_URL = "https://ai.hatz.ai/v1"

_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


class HatzAIError(Exception):
    pass


class HatzAIClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ["HATZAI_API_KEY"]
        self.model = model or os.environ.get("HATZAI_MODEL", "anthropic.claude-sonnet-4-6")
        self._session = requests.Session()
        self._session.headers.update({
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        })

    def list_models(self) -> list[dict]:
        resp = self._session.get(f"{BASE_URL}/chat/models")
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", data) if isinstance(data, dict) else data

    def chat(
        self,
        user_message: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        max_retries: int = 3,
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

        last_error: Exception = HatzAIError("Unknown error")

        for attempt in range(max_retries):
            try:
                resp = self._session.post(f"{BASE_URL}/chat/completions", json=payload)

                if resp.ok:
                    data = resp.json()
                    try:
                        return data["choices"][0]["message"]["content"]
                    except (KeyError, IndexError) as e:
                        raise HatzAIError(f"Unexpected response shape: {data}") from e

                if resp.status_code in _RETRY_STATUS_CODES:
                    wait = 2 ** attempt        # 1s, 2s, 4s
                    last_error = HatzAIError(
                        f"HatzAI API error {resp.status_code} (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)
                    continue

                # non-retryable error
                raise HatzAIError(f"HatzAI API error {resp.status_code}: {resp.text}")

            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt
                last_error = e
                time.sleep(wait)

        raise HatzAIError(f"Failed after {max_retries} attempts: {last_error}") from last_error

    def chat_json(
        self,
        user_message: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> dict:
        """Chat and parse the response as JSON. Retries once if parsing fails."""
        raw = self.chat(user_message, system=system, temperature=temperature,
                        max_tokens=max_tokens).strip()

        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            fix_prompt = f"Return ONLY valid JSON with no explanation or markdown:\n{raw}"
            fixed = self.chat(fix_prompt, temperature=0.0, max_tokens=max_tokens).strip()
            if fixed.startswith("```"):
                lines = fixed.splitlines()
                fixed = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return json.loads(fixed)
