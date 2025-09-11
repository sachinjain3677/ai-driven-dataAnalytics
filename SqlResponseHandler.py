import json
import duckdb
import sqlglot
from langfuse import get_client
from LLMResponseGenerator import call_llm

langfuse = get_client()

# Reuse the persisted database
conn = duckdb.connect('my_data.duckdb')

# Send prompt to LLM to get response
def get_sql_query_from_llm(prompt: str) -> str:
    raw_response = call_llm(prompt, span_name="ollama_generate_sql", external_id="request_12345")

    try:
        parsed = json.loads(raw_response)
        sql_query = parsed.get("sql_query")
        if not sql_query:
            raise ValueError("SQL query key not found in LLM response JSON.")
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse LLM response as JSON:\n{raw_response}")

    return sql_query


# # Parse the LLM response received and collect the required SQL query
# def parse_llm_response(full_answer: str) -> str:
#     try:
#         parsed = json.loads(full_answer)
#         generated_sql = parsed.get("sql_query")
#         print("\nGenerated SQL query: " + generated_sql )
#         if not generated_sql:
#             raise ValueError("Missing 'sql_query' in LLM response.")
#         return generated_sql
#     except json.JSONDecodeError:
#         raise ValueError("Failed to parse LLM response as JSON.")


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