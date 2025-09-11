import json
import plotly.express as px
import requests
import os
from langfuse import get_client

# Set Langfuse credentials
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-391f193d-c128-4eb7-a2a4-643fdccb6fa7"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-9c772c3f-4c63-4887-8df0-41e88747854c"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

langfuse = get_client()

# Get LLM to decide what kind of graph to be made for the usecase
def get_graph_metadata_ollama_multitable(prompt):
    """
    Calls the locally hosted Mistral 7B model via Ollama API to generate graph metadata.
    Returns a dict with keys: graph_type, x, y, title
    """
    full_answer = ""
    external_id = "graph_request_12345"
    trace_id = langfuse.create_trace_id(seed=external_id)

    with langfuse.start_as_current_span(
            name="ollama_generate_graph",
            input={"prompt": prompt},
            trace_context={"trace_id": trace_id}
    ) as span:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "temperature": 0.5},
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

        # If empty response, raise error
        if not full_answer.strip():
            raise ValueError("LLM returned empty response. Reduce prompt size or increase temperature.")

        # Parse JSON directly
        try:
            metadata = json.loads(full_answer.strip())
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse JSON from model output:\n{full_answer.strip()}")

        span.update(output={"graph_metadata": metadata})

        print("\n Full Answer of graph prompt : ", metadata)

    return metadata

# Plot graph according to result by LLM
def plot_graph(df, metadata):
    graph_type = metadata["graph_type"].lower()
    x_col = metadata["x"]
    y_col = metadata.get("y", None)
    title = metadata.get("title", "")

    if graph_type == "line":
        fig = px.line(df, x=x_col, y=y_col, title=title)
    elif graph_type == "bar":
        fig = px.bar(df, x=x_col, y=y_col, title=title)
    elif graph_type == "scatter":
        fig = px.scatter(df, x=x_col, y=y_col, title=title)
    elif graph_type == "pie":
        fig = px.pie(df, names=x_col, values=y_col, title=title)
    elif graph_type == "histogram":
        fig = px.histogram(df, x=x_col, title=title)
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    fig.show()
    return fig