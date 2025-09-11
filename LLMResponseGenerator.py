import requests
from langfuse import get_client
import json
import os

# Set Langfuse credentials
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-391f193d-c128-4eb7-a2a4-643fdccb6fa7"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-9c772c3f-4c63-4887-8df0-41e88747854c"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

langfuse = get_client()

# Function to call LLM with some specifications and get a raw response
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
    trace_id = langfuse.create_trace_id(seed=external_id)

    with langfuse.start_as_current_span(
            name=span_name,
            input={"prompt": prompt},
            trace_context={"trace_id": trace_id}
    ) as span:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt},
            stream=True
        )

        for line in response.iter_lines():
            if line:
                raw = line.decode("utf-8")
                try:
                    data = json.loads(raw)
                    if "response" in data:
                        full_answer += data["response"]
                except json.JSONDecodeError:
                    span.update(metadata={"json_decode_error": raw})

        span.update(output={"llm_response": full_answer.strip()})

    return full_answer.strip()
