import requests
import json
import os
from tracing import tracer

# Function to call LLM with some specifications and get a raw response
@tracer.chain()
def call_llm(prompt: str, span_name: str, external_id: str) -> str:
    """
    Generic function to call the locally hosted Mistral model via Ollama API.

    Args:
        prompt (str): Prompt to send to the LLM.
        span_name (str): Langfuse trace span name.
        external_id (str): Unique external trace ID seed.

    Returns:
        str: Raw LLM response string.
    """

    full_answer = ""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt},
            stream=True
        )
        print(f"[INFO] Request sent to LLM API, status_code={response.status_code}")

        for line in response.iter_lines():
            if line:
                raw = line.decode("utf-8")
                try:
                    data = json.loads(raw)
                    if "response" in data:
                        full_answer += data["response"]
                except json.JSONDecodeError:
                    print(f"[WARN] JSON decode error for line: {raw}")

        # print(f"[INFO] LLM raw response received. Length={len(full_answer)}")

    except Exception as e:
        print(f"[EXCEPTION] LLM call failed: {type(e).__name__}: {e}")
        raise

    return full_answer.strip()

