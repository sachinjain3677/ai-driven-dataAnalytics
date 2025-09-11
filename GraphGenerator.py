import json
import plotly.express as px
from LLMResponseGenerator import call_llm

# Get LLM to decide what kind of graph to be made for the usecase
def get_graph_metadata_from_llm(prompt: str) -> dict:
    raw_response = call_llm(prompt, span_name="ollama_generate_graph", external_id="graph_request_12345")

    try:
        metadata = json.loads(raw_response)
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse graph metadata JSON:\n{raw_response}")

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