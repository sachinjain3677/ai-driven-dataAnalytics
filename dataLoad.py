### 1. Load data (ETL service - extract, transform, load) and save schema and sample to refer later

# dataLoad.py
import pandas as pd
import duckdb
import json
from datetime import datetime
from typing import Dict

# Import the LLM utility from your other file
from LLMResponseGenerator import call_llm  # adjust import path as needed
from LLMPrompts import generate_dtype_prompt
from phoenixHelper import *

# -------------------------
# Global variables
# -------------------------
conn = None
orders_df = None
employees_df = None
customers_df = None
date_format = "%m/%d/%Y %I:%M:%S %p"  # adjust to match your CSV format - assumption for user csv data format

orders_csv_path = "dataset/orders.csv"
employees_csv_path = "dataset/employees.csv"
customers_csv_path = "dataset/customers.csv"

# -------------------------
# Database connection
# -------------------------
@tracer.chain()
def connect_to_duckdb(db_path='my_data.duckdb'):
    global conn
    if conn is None:
        print("Connecting to DuckDB...")
        conn = duckdb.connect(db_path)
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
    print("prompt given for", table_name, ":", prompt)
    llm_response = call_llm(prompt, span_name=f"infer_types_{table_name}", external_id=table_name)

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

# Global dictionary to persist column -> original dtype mapping
GLOBAL_DTYPE_DICT = {}

@tracer.chain()
def parse_dtype_dict_to_pandas_dtypes(dtype_dict: dict) -> tuple[dict, list]:
    """
    Converts a dictionary of column_name -> dtype (from LLM or other source)
    into a pandas dtype mapping and list of datetime columns.
    Also stores the original column -> dtype mapping in GLOBAL_DTYPE_DICT
    for future reference.

    Args:
        dtype_dict (dict): {column_name: dtype_string}

    Returns:
        pandas_dtype_map (dict): column -> pandas dtype (Int64, float, string)
        parse_dates (list): list of columns to parse as datetime
    """
    global GLOBAL_DTYPE_DICT
    print(f"[INFO] Parsing dtype dictionary: {dtype_dict}")

    # Persist original mapping
    GLOBAL_DTYPE_DICT.update(dtype_dict)
    print(f"[INFO] Updated GLOBAL_DTYPE_DICT: {GLOBAL_DTYPE_DICT}")

    parse_dates = []
    pandas_dtype_map = {}

    for col, dtype_val in dtype_dict.items():
        dtype_val_lower = dtype_val.lower()
        print(f"[DEBUG] Processing column '{col}' with dtype '{dtype_val_lower}'")

        if dtype_val_lower in ["int", "integer"]:
            pandas_dtype_map[col] = "Int64"
            print(f"[INFO] Mapped column '{col}' -> Int64")
        elif dtype_val_lower in ["float", "double"]:
            pandas_dtype_map[col] = "float"
            print(f"[INFO] Mapped column '{col}' -> float")
        elif dtype_val_lower in ["str", "string", "object"]:
            pandas_dtype_map[col] = "string"
            print(f"[INFO] Mapped column '{col}' -> string")
        elif dtype_val_lower in ["datetime", "datetime64"]:
            parse_dates.append(col)
            print(f"[INFO] Column '{col}' marked for datetime parsing")
        else:
            pandas_dtype_map[col] = "string"
            print(f"[WARN] Unknown dtype '{dtype_val}' for column '{col}', defaulting to string")

    print(f"[INFO] Final pandas dtype map: {pandas_dtype_map}")
    print(f"[INFO] Datetime columns: {parse_dates}")

    return pandas_dtype_map, parse_dates



# -------------------------
# Load CSV with LLM-inferred dtypes and return a DataFrame
# -------------------------
def print_dict_items(d: dict[str, str]) -> None:
    """Prints key-value pairs from a dictionary."""
    for key, value in d.items():
        print(f"{key}: {value}")

@tracer.chain()
def load_csv_with_llm_dtypes(csv_path: str, table_name: str) -> pd.DataFrame:
    # Step 1: Call LLM to infer datatypes
    dtype_dict = infer_dtypes_from_csv(csv_path, table_name)
    print(f"dtypes fetched for table: {table_name} from llm")
    print_dict_items(dtype_dict)

    # Step 2: Parse LLM response into pandas dtype mapping
    pandas_dtype_map, parse_dates = parse_dtype_dict_to_pandas_dtypes(dtype_dict)

    # Step 3: Read CSV using inferred dtypes
    df = pd.read_csv(
        csv_path,
        dtype=pandas_dtype_map,
        parse_dates=parse_dates,
        date_parser=lambda x: (
            datetime.strptime(str(x), "%m/%d/%Y %I:%M:%S %p") if pd.notnull(x) else pd.NaT
        )
    )

    # Step 4: Clean column names
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    print(table_name, "\n", df.head())
    return df


# -------------------------
# Helper to map pandas dtype to DuckDB type
# -------------------------
def pandas_dtype_to_duckdb(dtype: str) -> str:
    dtype = dtype.lower()
    if dtype in ["int64", "int", "integer"]:
        return "INTEGER"
    elif dtype in ["float64", "float", "double"]:
        return "DOUBLE"
    elif dtype in ["string", "object"]:
        return "VARCHAR"
    elif dtype in ["datetime64[ns]", "datetime"]:
        return "TIMESTAMP"
    return "VARCHAR"  # fallback

# -------------------------
# DuckDB table helpers
# -------------------------
def table_exists(con, table_name: str) -> bool:
    result = con.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = '{table_name}'
    """).fetchone()[0]
    return result > 0

# -------------------------
# Create table in DuckDB using DataFrame and inferred dtypes
# -------------------------
@tracer.chain()
def create_table_with_llm_dtypes(df: pd.DataFrame, table_name: str):
    if table_exists(conn, table_name):
        print(f"Table '{table_name}' already exists, dropping from db")
        conn.execute(f"DROP TABLE {table_name}")

    # Use the DataFrame's dtypes (already LLM-inferred) to map to DuckDB types
    print("printing col and dtypes for table: ", table_name)
    col_dtypes = [f"{col} {dtype}" for col, dtype in df.dtypes.items()]
    print(col_dtypes)
    cols = [f"{col} {pandas_dtype_to_duckdb(str(dtype))}" for col, dtype in df.dtypes.items()]
    ddl = f"CREATE TABLE {table_name} ({', '.join(cols)})"
    conn.execute(ddl)

    # Insert data
    conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
    print(f"Table '{table_name}' created with LLM-inferred datatypes.")

# -------------------------
# Load all CSVs into DuckDB with LLM-inferred dtypes
# -------------------------
@tracer.chain()
def load_data_into_duckdb_with_llm():
    global orders_df, employees_df, customers_df
    global orders_csv_path, employees_csv_path, customers_csv_path
    print("Loading CSVs into DuckDB with LLM-inferred datatypes...")

    # Step 1: Load CSVs using LLM dtypes
    orders_df = load_csv_with_llm_dtypes(orders_csv_path, "orders")
    employees_df = load_csv_with_llm_dtypes(employees_csv_path, "employees")
    customers_df = load_csv_with_llm_dtypes(customers_csv_path, "customers")

    # Step 2: Create tables in DuckDB using LLM-inferred dtypes
#     create_table_with_llm_dtypes(orders_df, "orders")
#     create_table_with_llm_dtypes(employees_df, "employees")
#     create_table_with_llm_dtypes(customers_df, "customers")

# -------------------------
# Utility functions: fetch schemas and sample rows
# -------------------------
@tracer.chain()
def get_schemas() -> dict:
    """Return inferred schema/datatypes for all three dataframes."""
    global orders_df, employees_df, customers_df
    return {
        "orders": {col: str(dtype) for col, dtype in orders_df.dtypes.items()},
        "employees": {col: str(dtype) for col, dtype in employees_df.dtypes.items()},
        "customers": {col: str(dtype) for col, dtype in customers_df.dtypes.items()}
    }

def get_top_rows(n: int = 5) -> dict:
    """Return top n rows from all three dataframes as list of dicts."""
    global orders_df, employees_df, customers_df
    return {
        "orders": orders_df.head(n).to_dict(orient="records"),
        "employees": employees_df.head(n).to_dict(orient="records"),
        "customers": customers_df.head(n).to_dict(orient="records")
    }




### 2. Call LLM with user query, schema and sample data - hugging face LLm Qwen

# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#
# # Load public model
# tokenizer = AutoTokenizer.from_pretrained("Ellbendls/Qwen-2.5-3b-Text_to_SQL")
# model = AutoModelForCausalLM.from_pretrained("Ellbendls/Qwen-2.5-3b-Text_to_SQL")
#
# # Take user input
# user_query = "Find the names of all the companies that had their orders shipped to France"
# #user_query = input("Enter your natural language query: ")
#
# # Construct prompt
# prompt = f"""
# You are a SQL generation assistant.
#
# Schema:
# {schema_text}
#
# Sample Data:
# {samples_text}
#
# User Query:
# {user_query}
#
# Generate the appropriate SQL query to retrieve the requested data.
# """
#
# # Tokenize input and generate output
# print("\n Tokenizing")
# inputs = tokenizer(prompt, return_tensors="pt")
#
# # Start clock
# start_time = time.time()
#
# print("\n Hang on .. model is Generating")
# outputs = model.generate(**inputs, max_new_tokens=100)
#
# # Stop clock
# end_time = time.time()
# elapsed_time = end_time - start_time
#
# # Decode and print the generated SQL
# print("\n Decoding the numbers for you ")
# generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("\nGenerated SQL:\n", generated_sql)
#
# # Print elapsed time
# print(f"\nTime taken for generation: {elapsed_time:.2f} seconds")