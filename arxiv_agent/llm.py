import json
import os
import re
import time

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

from .config import Settings

REASONING_MODELS = ("gpt-oss", "qwen3", "deepseek-r1")
REASONING_HEADROOM = 1500


class LLMError(RuntimeError):
    pass


def parse_json_object(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^<think>.*?</think>\s*", "", raw, flags=re.S)
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.S)
    if fenced:
        raw = fenced.group(1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise ValueError("model output did not contain a JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model output was JSON but not an object")
    return value


def _retry_after(exc: RateLimitError) -> float | None:
    try:
        return float(exc.response.headers.get("retry-after"))
    except (TypeError, ValueError, AttributeError):
        return None


class LLM:
    def __init__(self, settings: Settings, max_attempts: int = 5):
        if settings.provider == "groq" and not settings.api_key:
            raise LLMError("GROQ_API_KEY is missing. Copy .env.example to .env and paste your key.")
        self.model = settings.model
        self.base_url = settings.base_url
        self.max_attempts = max_attempts
        self.reasoning = any(tag in self.model.lower() for tag in REASONING_MODELS)
        self.reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "low")
        self.client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key or "not-needed",
            timeout=120,
            max_retries=0,
        )

    def _request(self, messages: list[dict], max_tokens: int, temperature: float, json_mode: bool):
        extra: dict = {}
        if json_mode:
            extra["response_format"] = {"type": "json_object"}
        if self.reasoning:
            max_tokens += REASONING_HEADROOM
            if "gpt-oss" in self.model.lower():
                extra["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **extra,
        )

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 1200,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        delay = 4.0
        grew = False
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._request(messages, max_tokens, temperature, json_mode)
                choice = response.choices[0]
                content = re.sub(r"^<think>.*?</think>\s*", "", choice.message.content or "", flags=re.S).strip()
                if not content and choice.finish_reason == "length" and not grew:
                    max_tokens *= 2
                    grew = True
                    continue
                if not content:
                    raise LLMError("the model returned an empty answer")
                return content
            except RateLimitError as exc:
                if attempt == self.max_attempts:
                    raise LLMError("rate limit did not clear; wait a minute and retry") from exc
                wait = _retry_after(exc) or delay
                print(f"  rate limited, waiting {wait:.0f}s")
                time.sleep(wait)
                delay = min(delay * 2, 60)
            except NotFoundError as exc:
                raise LLMError(
                    f"model '{self.model}' is not available on {self.base_url}. "
                    "Set LLM_MODEL in .env to a current model (see README)."
                ) from exc
            except AuthenticationError as exc:
                raise LLMError("the API key was rejected. Check GROQ_API_KEY in .env.") from exc
            except BadRequestError as exc:
                if json_mode:
                    json_mode = False
                    continue
                raise LLMError(f"LLM rejected the request: {exc}") from exc
            except (APIConnectionError, APITimeoutError) as exc:
                if attempt == self.max_attempts:
                    raise LLMError(f"could not reach the LLM at {self.base_url}") from exc
                time.sleep(delay)
                delay = min(delay * 2, 60)
            except APIStatusError as exc:
                if exc.status_code >= 500 and attempt < self.max_attempts:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise LLMError(f"LLM request failed with HTTP {exc.status_code}: {exc}") from exc
        raise LLMError("LLM call failed")

    def chat_json(self, system: str, user: str, max_tokens: int = 1500) -> dict:
        raw = self.chat(system, user, max_tokens=max_tokens, temperature=0.1, json_mode=True)
        return parse_json_object(raw)
