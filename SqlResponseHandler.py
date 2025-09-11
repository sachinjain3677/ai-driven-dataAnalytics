import json
import duckdb
import sqlglot
import os
import requests
from langfuse import get_client

# Set Langfuse credentials
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-391f193d-c128-4eb7-a2a4-643fdccb6fa7"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-9c772c3f-4c63-4887-8df0-41e88747854c"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

langfuse = get_client()

# Reuse the persisted database
conn = duckdb.connect('my_data.duckdb')

# Send prompt to LLM to get response
def get_llm_response(prompt: str) -> str:
    full_answer = ""
    external_id = "request_12345"
    trace_id = langfuse.create_trace_id(seed=external_id)

    with langfuse.start_as_current_span(
            name="ollama_generate",
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

        span.update(output={"generated_sql": full_answer.strip()})

    return full_answer.strip()


# Parse the LLM response received and collect the required SQL query
def parse_llm_response(full_answer: str) -> str:
    try:
        parsed = json.loads(full_answer)
        generated_sql = parsed.get("sql_query")
        print("\nGenerated SQL query: " + generated_sql )
        if not generated_sql:
            raise ValueError("Missing 'sql_query' in LLM response.")
        return generated_sql
    except json.JSONDecodeError:
        raise ValueError("Failed to parse LLM response as JSON.")


# Validate sql received from LLM response using sqlglot
def validate_and_normalize_sql(sql_query: str) -> str:
    try:
        # Parse the SQL (no need to specify dialect for parsing)
        expression = sqlglot.parse_one(sql_query)

        # Ensure it is a SELECT query
        if not isinstance(expression, sqlglot.expressions.Select):
            raise ValueError("Only SELECT queries are allowed.")

        # Convert back to normalized SQL string for DuckDB
        normalized_sql = expression.sql(dialect="duckdb")

        return normalized_sql

    except Exception as e:
        raise ValueError(f"SQL validation failed: {str(e)}")


# Executes the validated sql on db
def execute_sql(sql_query: str) -> list:
    try:
        # Execute the query on the existing DuckDB connection
        result = conn.execute(sql_query).fetchall()

        # Get column names
        columns = [desc[0] for desc in conn.description]

        # Convert result into list of dicts
        result_dicts = [dict(zip(columns, row)) for row in result]
        return result_dicts

    except Exception as e:
        raise ValueError(f"SQL execution failed: {str(e)}")