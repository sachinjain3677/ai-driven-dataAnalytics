### 1. Load data (ETL service - extract, transform, load) and save schema and sample to refer later

# dataLoad.py
import pandas as pd
import duckdb

# Global variables to hold connection and DataFrames
conn = None
orders_df = None
employees_df = None
customers_df = None

# -------------------------
# Database connection
# -------------------------
def connect_to_duckdb(db_path='my_data.duckdb'):
    global conn
    if conn is None:
        print("Connecting to DuckDB...")
        conn = duckdb.connect(db_path)
    return conn

# -------------------------
# Load CSVs into DataFrames
# -------------------------
def load_csvs():
    global orders_df, employees_df, customers_df
    print("Loading data from CSVs...")
    orders_df = pd.read_csv("orders.csv")
    employees_df = pd.read_csv("employees.csv")
    customers_df = pd.read_csv("customers.csv")
    return orders_df, employees_df, customers_df

# -------------------------
# Clean column names
# -------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    return df

def clean_loaded_data():
    global orders_df, employees_df, customers_df
    print("Cleaning loaded data...")
    orders_df = clean_columns(orders_df)
    employees_df = clean_columns(employees_df)
    customers_df = clean_columns(customers_df)
    return orders_df, employees_df, customers_df

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

def create_table_if_not_exists(df: pd.DataFrame, table_name: str):
    global conn
    if not table_exists(conn, table_name):
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        print(f"Table '{table_name}' created.")
    else:
        print(f"Table '{table_name}' already exists.")

def load_data_into_duckdb():
    global orders_df, employees_df, customers_df, conn
    print("Loading tables into DuckDB...")
    create_table_if_not_exists(orders_df, "orders")
    create_table_if_not_exists(employees_df, "employees")
    create_table_if_not_exists(customers_df, "customers")

# -------------------------
# Schema & sample helpers
# -------------------------
def build_schema_text(df: pd.DataFrame, table_name: str) -> str:
    lines = [f"Table {table_name}:"]
    for col, dtype in df.dtypes.items():
        lines.append(f"  - {col}: {str(dtype)}")
    return "\n".join(lines)

def sample_rows_text(df: pd.DataFrame, n=1) -> str:
    samples = df.head(n).to_dict(orient="records")
    return "\n".join([str(r) for r in samples])

def get_schema_text() -> str:
    global orders_df, employees_df, customers_df
    return (
        build_schema_text(orders_df, "orders") + "\n\n" +
        build_schema_text(employees_df, "employees") + "\n\n" +
        build_schema_text(customers_df, "customers")
    )

def get_samples_text() -> str:
    global orders_df, employees_df, customers_df
    return (
        sample_rows_text(orders_df) + "\n\n" +
        sample_rows_text(employees_df) + "\n\n" +
        sample_rows_text(customers_df)
    )





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
# print("\n Tokeninzing")
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