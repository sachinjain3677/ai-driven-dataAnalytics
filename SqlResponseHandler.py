import json
import duckdb
import sqlglot
from langfuse import get_client
from LLMResponseGenerator import call_llm
from LLMPrompts import generate_dtype_prompt
from dataLoad import generate_dtype_prompt, parse_llm_dtype_response, parse_dtype_dict_to_pandas_dtypes
import pandas as pd

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


def execute_sql(sql_query: str) -> pd.DataFrame:
    """
    Executes the validated SQL on DuckDB, infers datatypes using LLM,
    and returns the result as a DataFrame with appropriate dtypes.
    """
    try:
        # Execute the query
        result = conn.execute(sql_query).fetchall()

        # Get column names
        columns = [desc[0] for desc in conn.cursor().description]

        # Convert result into list of dicts
        result_dicts = [dict(zip(columns, row)) for row in result]

        # Prepare a sample row for LLM dtype inference
        sample_row = result_dicts[0] if result_dicts else {col: None for col in columns}

        # Generate prompt using external utility
        prompt = generate_dtype_prompt(sample_row, table_name="sql_result")

        # Call the LLM to infer datatypes
        llm_response = call_llm(prompt, span_name="infer_sql_result_types", external_id="sql_result")

        # Reuse helper to parse LLM response into pandas dtype map and datetime columns
        pandas_dtype_map, parse_dates = parse_dtype_dict_to_pandas_dtypes(llm_response)

        # Create DataFrame with inferred dtypes
        df = pd.DataFrame(result_dicts)
        for col, pd_dtype in pandas_dtype_map.items():
            df[col] = df[col].astype(pd_dtype)
        for col in parse_dates:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    except Exception as e:
        raise ValueError(f"SQL execution failed: {str(e)}")



