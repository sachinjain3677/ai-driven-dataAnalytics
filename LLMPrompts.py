import json

# -------------------------
# Prompt generation for LLM dtype inference
# -------------------------
def generate_dtype_prompt(first_row: dict, table_name: str) -> str:
    """Generate a prompt for the LLM to infer column datatypes from the first row."""
    return (
        f"Given the first row of a CSV for table '{table_name}', "
        f"suggest the most relevant Python/pandas datatype for each column. "
        f"Respond in the format 'column_name: datatype'.\n\n"
        f"Row sample: {json.dumps(first_row, indent=2)}"
    )

# Prompt to be given to LLM to generate SQL query
def get_sql_prompt(schema_text: str, samples_text: str, user_query: str) -> str:
    return f"""
    You are a SQL expert. Based on the schema below:
    {schema_text}
    
    Sample Data:
    {samples_text}
    
    Important instructions for DuckDB:
    - All columns are stored with correct datatypes (e.g., DATE, TIMESTAMP, INTEGER, VARCHAR).
    - You can directly use date/time functions (YEAR(), MONTH(), DAY(), comparisons) on DATE/TIMESTAMP columns.
    - Use ISO-standard date literals in the format 'YYYY-MM-DD' when filtering or comparing DATE/TIMESTAMP values.
    - Do not attempt to parse or cast columns like `orderdate` manually — they are already stored as proper DATE/TIMESTAMP types.
    - Always generate syntactically valid DuckDB SQL.

    Generate an appropriate DuckDB-compatible SQL query for this natural language request:
    {user_query}

    Respond **ONLY in this JSON format**:
    {{
        "sql_query": "SELECT ...",
        "explanation": "A brief explanation of the query"
    }}
    """


# LLM Prompt to predict graph type according to user query, sample data and dataset schema
def create_graph_prompt(schema_text: str, samples_text: str, user_query: str) -> str:
    prompt = f"""
        You are a data visualization expert. Your task is to determine the most appropriate graph for a user's request.

        Schema:
        {schema_text}

        Sample data:
        {samples_text}

        User query: "{user_query}"

        Instructions:
        - Select only one of these graph types exactly: "Line", "Bar", "Pie", "Scatter", "Histogram".
        - Use only column names from the schema provided.
        - Respond ONLY in valid JSON format, with these exact keys and types:

        {{
          "graph_type": "Line" | "Bar" | "Pie" | "Scatter" | "Histogram",
          "x": "string",
          "y": "string or null",
          "title": "string"
        }}

        - Do NOT provide any explanation, commentary, or anything else.
        - Do NOT use any graph type names other than the ones explicitly listed.
        - Do NOT use additional qualifiers like "Filtered Bar" or "Grouped Histogram".
        - Output ONLY the valid JSON object, with no code fences, no examples, nothing else.

        Example of correct output:

        {{
            "graph_type": "Bar",
            "x": "shipcountry",
            "y": "freight",
            "title": "Freight Details of Orders Shipped to Germany in 1995"
        }}
        """
    return prompt
