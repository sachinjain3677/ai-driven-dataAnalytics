import json
from fastapi import FastAPI, Request
from dataLoad import get_schema_text, get_samples_text
from LLMPrompts import *
from SqlResponseHandler import *
from GraphGenerator import *

app = FastAPI()

@app.post("/generate_sql")
async def generate_sql(request: Request):
    body = await request.json()
    user_query = body.get("query")  # Natural language query

    # Define schema_text and samples_text (hardcoded for now)
    schema_text = get_schema_text()
    samples_text = get_samples_text()

    # Build the full prompt
    prompt = get_sql_prompt(schema_text, samples_text, user_query)
    generated_sql = get_sql_query_from_llm(prompt)

    try:
        print("\n generated sql : " + generated_sql)
        validated_sql = validate_and_normalize_sql(generated_sql)
        query_results = execute_sql(validated_sql)

        print("\n Fetching kind of graph to plot")
        # Create prompt for getting graph type and axis details
        graph_prompt = create_graph_prompt(schema_text, samples_text, user_query)
        print("\n Graph prompt : " + graph_prompt)
        # Get graph metadata
        metadata = get_graph_metadata_from_llm(graph_prompt)
        # Plot the graph
        print("\n Plotting ... ")
        fig = plot_graph(query_results, metadata)

    except Exception as e:
        return {"error": str(e)}
