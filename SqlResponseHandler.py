import json
import duckdb
import sqlglot
from LLMResponseGenerator import call_llm
from LLMPrompts import generate_dtype_prompt
from dataLoad import generate_dtype_prompt, parse_llm_dtype_response, parse_dtype_dict_to_pandas_dtypes, GLOBAL_DTYPE_DICT
import pandas as pd
import inspect
from tracing import tracer


# Reuse the persisted database
conn = duckdb.connect('my_data.duckdb')

# Send prompt to LLM to get response
@tracer.chain()
def get_sql_query_from_llm(prompt: str) -> str:
    print("[INFO] LLM called from: ", inspect.currentframe().f_code.co_name)
    raw_response = call_llm(prompt)

    try:
        parsed = json.loads(raw_response)
        sql_query = parsed.get("sql_query")
        if not sql_query:
            raise ValueError("SQL query key not found in LLM response JSON.")
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse LLM response as JSON:\n{raw_response}")

    return sql_query

# Validate sql received from LLM response using sqlglot
@tracer.chain()
def validate_and_normalize_sql(sql_query: str) -> str:
    try:
        print(f"[INFO] Validating SQL query: {sql_query}")

        # Parse the SQL (no need to specify dialect for parsing)
        expression = sqlglot.parse_one(sql_query)
        print(f"[DEBUG] Parsed SQL expression: {expression}")

        # Ensure it is a SELECT query
        if not isinstance(expression, sqlglot.expressions.Select):
            print(f"[ERROR] Non-SELECT query detected: {sql_query}")
            raise ValueError("Only SELECT queries are allowed.")

        # Convert back to normalized SQL string for DuckDB
        normalized_sql = expression.sql(dialect="duckdb")
        print(f"[INFO] Normalized SQL for DuckDB: {normalized_sql}")

        return normalized_sql

    except Exception as e:
        print(f"[EXCEPTION] SQL validation failed: {e}")
        raise ValueError(f"SQL validation failed: {str(e)}")



@tracer.chain()
def execute_sql(sql_query: str) -> pd.DataFrame:
    """
    Executes the validated SQL on DuckDB, applies datatypes from GLOBAL_DTYPE_DICT,
    and returns the result as a DataFrame with appropriate dtypes.
    """
    try:
        print(f"[INFO] Executing SQL query: {sql_query}")

        # Execute the query
        cursor = conn.execute(sql_query)
        result = cursor.fetchall()
        print(f"[INFO] Query executed successfully. Rows fetched: {len(result)}")

        # Get column names
        columns = [desc[0] for desc in cursor.description]
        print(f"[DEBUG] Columns fetched: {columns}")

        # Convert result into list of dicts
        result_dicts = [dict(zip(columns, row)) for row in result]
        print(f"[DEBUG] First row sample: {result_dicts[0] if result_dicts else 'No rows'}")

        # Use global dtype mapping instead of calling LLM
        print(f"[INFO] Using GLOBAL_DTYPE_DICT for dtype inference: {GLOBAL_DTYPE_DICT}")

        pandas_dtype_map, parse_dates = parse_dtype_dict_to_pandas_dtypes(GLOBAL_DTYPE_DICT)
        print(f"[DEBUG] Pandas dtype map: {pandas_dtype_map}")
        print(f"[DEBUG] Datetime columns: {parse_dates}")

        # Create DataFrame with inferred dtypes
        df = pd.DataFrame(result_dicts)
        for col, pd_dtype in pandas_dtype_map.items():
            if col in df.columns:
                print(f"[INFO] Casting column '{col}' to dtype '{pd_dtype}'")
                df[col] = df[col].astype(pd_dtype)
        for col in parse_dates:
            if col in df.columns:
                print(f"[INFO] Parsing column '{col}' as datetime")
                df[col] = pd.to_datetime(df[col], errors='coerce')

        print(f"[INFO] DataFrame created successfully with {len(df)} rows and {len(df.columns)} columns")
        return df

    except Exception as e:
        print(f"[EXCEPTION] SQL execution failed: {type(e).__name__}: {e}")
        raise ValueError(f"SQL execution failed: {str(e)}")





