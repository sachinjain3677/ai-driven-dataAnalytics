### 1. Load data (ETL service - extract, transform, load) and save schema and sample to refer later

# dataLoad.py
import pandas as pd
import duckdb
from datetime import datetime
from typing import Dict
import os

# Import the LLM utility from your other file
from LLMResponseGenerator import call_llm_infer_dtypes  # adjust import path as needed
from LLMPrompts import generate_dtype_prompt
import inspect
from tracing import tracer

from llm_as_a_judge.judgeHandler import judge_response_with_gemini
from redis_client import redis_client

llm_as_a_judge = os.getenv("LLM_AS_A_JUDGE", "false").lower() in ("true", "1", "yes")

# -------------------------
# Global variables
# -------------------------
conn = None
date_format = "%m/%d/%Y %I:%M:%S %p"  # adjust to match your CSV format - assumption for user csv data format

# -------------------------
# Database connection
# -------------------------
@tracer.chain()
def connect_to_duckdb(db_path='my_data.duckdb'):
    global conn
    if conn is None:
        print("Connecting to DuckDB...")
        conn = duckdb.connect(db_path, read_only=False)
    return conn

# -------------------------
# Infer datatypes directly from CSV first row
# -------------------------
def get_first_row_from_csv(csv_path: str) -> Dict:
    """Read only the first row from a CSV as a dictionary."""
    df_sample = pd.read_csv(csv_path, nrows=1)
    return df_sample.to_dict(orient="records")[0]

@tracer.chain()
def infer_dtypes_from_csv(csv_path: str, table_name: str) -> Dict[str, str]:
    """Use LLM to infer Python/pandas datatypes for CSV columns."""
    first_row = get_first_row_from_csv(csv_path)
    prompt = generate_dtype_prompt(first_row, table_name)
    print("\n\n[INFO] LLM called from: ", inspect.currentframe().f_code.co_name, ", csv: ", csv_path)

    llm_response = call_llm_infer_dtypes(prompt)

    if (llm_as_a_judge):
        judge_response_with_gemini("dtypes", prompt, llm_response)

    return parse_llm_dtype_response(llm_response)

# -------------------------
# Utility to parse LLM dtype response
# -------------------------
@tracer.chain()
def parse_llm_dtype_response(llm_response: str) -> dict:
    """
    Parse the LLM response for datatypes into a dictionary of column -> dtype string.
    """
    dtype_mapping_dict = {}
    for line in llm_response.split("\n"):
        if ":" in line:
            col, dtype = line.split(":", 1)
            dtype_mapping_dict[col.strip()] = dtype.strip()
    return dtype_mapping_dict

@tracer.chain()
def parse_dtype_dict_to_pandas_dtypes(dtype_dict: dict) -> tuple[dict, list]:
    """
    Converts a dictionary of column_name -> dtype (from LLM or other source)
    into a pandas dtype mapping and list of datetime columns.
    Also stores the original column -> dtype mapping in a Redis hash.

    Args:
        dtype_dict (dict): {column_name: dtype_string}

    Returns:
        pandas_dtype_map (dict): column -> pandas dtype (Int64, float, string)
        parse_dates (list): list of columns to parse as datetime
    """
    # Persist the original mapping in a Redis hash
    redis_client.hset("dtype_cache", mapping=dtype_dict)
    print(f"[INFO] Updated Redis dtype_cache with: {dtype_dict}")

    parse_dates = []
    pandas_dtype_map = {}

    for col, dtype_val in dtype_dict.items():
        dtype_val_lower = dtype_val.lower()

        if dtype_val_lower in ["int", "integer"]:
            pandas_dtype_map[col] = "Int64"
        elif dtype_val_lower in ["float", "double"]:
            pandas_dtype_map[col] = "float"
        elif dtype_val_lower in ["str", "string", "object"]:
            pandas_dtype_map[col] = "string"
        elif dtype_val_lower in ["datetime", "datetime64"]:
            parse_dates.append(col)
        else:
            pandas_dtype_map[col] = "string"

    return pandas_dtype_map, parse_dates

# -------------------------
# Load CSV with LLM-inferred dtypes and return a DataFrame
# -------------------------
@tracer.chain()
def load_csv_with_llm_dtypes(csv_path: str, table_name: str) -> pd.DataFrame:
    dtype_dict = infer_dtypes_from_csv(csv_path, table_name)
    pandas_dtype_map, parse_dates = parse_dtype_dict_to_pandas_dtypes(dtype_dict)

    df = pd.read_csv(
        csv_path,
        dtype=pandas_dtype_map,
        parse_dates=parse_dates,
        date_format=date_format
    )

    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    return df

# -------------------------
# Helper to map pandas dtype to DuckDB type
# -------------------------
def pandas_dtype_to_duckdb(dtype: str) -> str:
    dtype = str(dtype).lower()
    if "int" in dtype:
        return "INTEGER"
    elif "float" in dtype:
        return "DOUBLE"
    elif "datetime" in dtype:
        return "TIMESTAMP"
    return "VARCHAR"  # fallback for object, string, etc.

# -------------------------
# DuckDB table helpers
# -------------------------
def table_exists(con, table_name: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
        return True
    except duckdb.Error:
        return False

# -------------------------
# Create table in DuckDB using DataFrame and inferred dtypes
# -------------------------
@tracer.chain()
def create_table_with_llm_dtypes(df: pd.DataFrame, table_name: str):
    global conn
    if table_exists(conn, table_name):
        conn.execute(f"DROP TABLE {table_name}")

    cols = [f'"{col}" {pandas_dtype_to_duckdb(str(dtype))}' for col, dtype in df.dtypes.items()]
    ddl = f'CREATE TABLE "{table_name}" ({', '.join(cols)})'
    conn.execute(ddl)

    conn.register('df_temp', df)
    conn.execute(f'INSERT INTO "{table_name}" SELECT * FROM df_temp')
    conn.unregister('df_temp')
    print(f"Table '{table_name}' created with LLM-inferred datatypes.")

# -------------------------
# Load a single CSV into DuckDB with LLM-inferred dtypes
# -------------------------
@tracer.chain()
def load_data_into_duckdb_with_llm(csv_path: str, table_name: str):
    print(f"Loading {csv_path} into DuckDB table {table_name} with LLM-inferred datatypes...")
    df = load_csv_with_llm_dtypes(csv_path, table_name)
    create_table_with_llm_dtypes(df, table_name)

# -------------------------
# Utility functions: fetch schemas and sample rows from DuckDB
# -------------------------
@tracer.chain()
def get_schemas() -> dict:
    """Return inferred schema/datatypes for all tables in DuckDB."""
    global conn
    schemas = {}
    tables_df = conn.execute("SHOW TABLES").fetchdf()
    for index, row in tables_df.iterrows():
        table_name = row['name']
        schema_df = conn.execute(f"DESCRIBE \"{table_name}\"").fetchdf()
        schemas[table_name] = dict(zip(schema_df['column_name'], schema_df['column_type']))
    return schemas

@tracer.chain()
def get_top_rows(n: int = 5) -> dict:
    """Return top n rows from all tables in DuckDB as a list of dicts."""
    global conn
    samples = {}
    tables_df = conn.execute("SHOW TABLES").fetchdf()
    for index, row in tables_df.iterrows():
        table_name = row['name']
        try:
            top_rows_df = conn.execute(f'SELECT * FROM "{table_name}" LIMIT {n}').fetchdf()
            samples[table_name] = top_rows_df.to_dict(orient="records")
        except duckdb.Error as e:
            print(f"Error fetching top rows for table {table_name}: {e}")
            samples[table_name] = []
    return samples

# -------------------------
# Reset functions
# -------------------------
@tracer.chain()
def reset_database():
    """Drops all tables from the DuckDB database."""
    global conn
    if conn is None:
        connect_to_duckdb()
    
    tables = conn.execute("SHOW TABLES").fetchall()
    if not tables:
        print("No tables to drop.")
        return

    for table in tables:
        table_name = table[0]
        print(f"Dropping table: {table_name}")
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    print("All tables have been dropped.")

def reset_dtype_cache():
    """Clears the dtype cache from Redis."""
    redis_client.delete("dtype_cache")
    print("Dtype cache has been cleared from Redis.")