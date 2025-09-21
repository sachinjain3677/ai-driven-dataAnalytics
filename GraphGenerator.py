import json
import pandas as pd
import plotly.express as px
from LLMResponseGenerator import call_llm_generate_graph
from plotly.graph_objs import Figure
import inspect
from tracing import tracer
import os
from llm_as_a_judge.judgeHandler import judge_response_with_gemini

llm_as_a_judge = os.getenv("LLM_AS_A_JUDGE", "false").lower() in ("true", "1", "yes")

# Get LLM to decide what kind of graph to be made for the usecase
@tracer.chain()
def get_graph_metadata_from_llm(prompt: str) -> dict:
    """
    Send a prompt to the LLM to decide graph metadata and parse the JSON response.

    Args:
        prompt (str): Prompt describing the graph requirements.

    Returns:
        dict: Graph metadata (graph_type, x, y, title, etc.)
    """
    print("[INFO] LLM called from: ", inspect.currentframe().f_code.co_name)
    raw_response = call_llm_generate_graph(prompt)

    if (llm_as_a_judge):
        judge_response_with_gemini("graph", prompt, raw_response)

    print(f"[INFO] Raw LLM response received (length={len(raw_response)} chars)")
    print(f"[DEBUG] Raw LLM response:\n{raw_response}")

    try:
        metadata = json.loads(raw_response)
        print("[INFO] Successfully parsed LLM response into metadata")
        print(f"[DEBUG] Parsed metadata: {metadata}")
    except json.JSONDecodeError as e:
        print(f"[EXCEPTION] Failed to parse LLM response as JSON: {e}")
        raise ValueError(f"Could not parse graph metadata JSON:\n{raw_response}")

    return metadata

# Plot graph according to result by LLM
@tracer.chain()
def plot_graph(df, metadata) -> Figure:
    """
    Plot a graph using Plotly based on LLM-provided metadata.

    Args:
        df (pd.DataFrame | list[dict]): Input data.
        metadata (dict): Instructions from LLM, e.g.
            {
                "graph_type": "bar",
                "x": "column_x",
                "y": "column_y",
                "title": "My Chart"
            }
    """
    print("[INFO] Starting plot_graph")


    # Save current options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # Print the entire DataFrame
    print(df)

    # Reset options to default (optional)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')

    print(f"[DEBUG] Metadata received: {metadata}")

    # Ensure df is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        print("[INFO] Converting input data to pandas DataFrame")
        df = pd.DataFrame(df)

    print(f"[DEBUG] DataFrame shape: {df.shape}")
    print(f"[DEBUG] DataFrame columns: {list(df.columns)}")

    graph_type = metadata["graph_type"].lower()
    x_col = metadata["x"]
    y_col = metadata.get("y", None)
    title = metadata.get("title", "")

    print(f"[INFO] Graph type: {graph_type}, X: {x_col}, Y: {y_col}, Title: '{title}'")

    # Select plot type
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
        print(f"[EXCEPTION] Unsupported graph type: {graph_type}")
        raise ValueError(f"Unsupported graph type: {graph_type}")

    print("[INFO] Plot created successfully, displaying graph")
    fig.show()
    return fig
