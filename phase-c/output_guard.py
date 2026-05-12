import requests
import time


class OutputGuardAPI:
    def __init__(self, api_key):
        self.api_key = api_key

        self.url = (
            "https://api.groq.com/openai/v1/chat/completions"
        )

    def check(self, user_input, agent_response):
        payload = {
            "model": "llama-guard-3-8b",
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                },
                {
                    "role": "assistant",
                    "content": agent_response,
                },
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        start = time.perf_counter()

        response = requests.post(
            self.url,
            json=payload,
            headers=headers,
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        result = response.json()["choices"][0][
            "message"
        ]["content"]

        is_safe = (
            "safe" in result.lower()
            and "unsafe" not in result.lower()
        )

        return is_safe, result, latency_ms