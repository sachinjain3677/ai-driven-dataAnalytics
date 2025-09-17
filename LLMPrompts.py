import json
from tracing import tracer

# -------------------------
# Prompt generation for LLM dtype inference
# -------------------------
@tracer.chain()
def generate_dtype_prompt(first_row: dict, table_name: str) -> str:
    """Generate a prompt for the LLM to infer column datatypes from the first row."""
    return (
        f"Given the first row of a CSV for table '{table_name}', "
        f"suggest the most relevant Python/pandas datatype for each column.\n\n"
        f"Important instructions:\n"
        f"- Respond ONLY with one column per line.\n"
        f"- Format: column_name: datatype\n"
        f"- Do not include explanations, extra text, or JSON.\n"
        f"- Use only these datatypes: int, float, string, datetime.\n\n"
        f"- datatype for POSTALCODE is always string\n\n"
        f"Row sample: {json.dumps(first_row, indent=2)}"
    )

# Prompt to be given to LLM to generate SQL query
@tracer.chain()
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
    - Only use columns that exist in the schema provided above.
    - Always generate syntactically valid DuckDB SQL.
    - Ensure the SELECT clause includes only columns from the schema.

    Generate an appropriate DuckDB-compatible SQL query for this natural language request:
    {user_query}

    Respond **ONLY in this JSON format**:
    {{
        "sql_query": "SELECT ...",
        "explanation": "A brief explanation of the query"
    }}
    """


# LLM Prompt to predict graph type according to user query, sample data and dataset schema
@tracer.chain()
def create_graph_prompt(schema_text: str, samples_text: str, user_query: str) -> str:
    prompt = f"""
        You are a data visualization expert. Your task is to determine the most appropriate graph for a user's request.

        Schema:
        {schema_text}

        Sample data:
        {samples_text}

        User query: "{user_query}"

        Instructions (STRICT):
        - If the user_query explicitly mentions a graph type (Line, Bar, Pie, Scatter, Histogram), give it highest priority.
        - Select only one of these graph types: "Line", "Bar", "Pie", "Scatter", "Histogram".
        - Use only column names from the schema provided.
        - Respond ONLY with a single valid JSON object, nothing else.
        - DO NOT include explanations, natural language, markdown, code fences, or comments.
        - DO NOT invent additional keys or change key names.
        - DO NOT add any prefixes like "Here is the JSON".
        - The output must be directly parsable by `json.loads`.
        - None of the columns (x or y) can be null or None, except for "Histogram" which only requires the "x" column.

        The JSON object must have exactly these keys:

        {{
          "graph_type": "Line" | "Bar" | "Pie" | "Scatter" | "Histogram",
          "x": "string",
          "y": "string or null",
          "title": "string"
        }}

        Example of correct output:

        {{
          "graph_type": "Bar",
          "x": "shipcountry",
          "y": "freight",
          "title": "Freight Details of Orders Shipped to Germany in 1995"
        }}
        """
    return prompt

# LLM Prompt to extract key insights from schema, sample data and user query
@tracer.chain()
def create_insight_prompt(schema_text: str, samples_text: str, user_query: str) -> str:
    prompt = f"""
        You are a data analysis expert. Your task is to find the most important insights from the given schema, data, and user query.

        Schema:
        {schema_text}

        Sample data:
        {samples_text}

        User query: "{user_query}"

        Instructions (STRICT):
        - Carefully analyze the schema and sample data in the context of the user query.
        - Identify only the most relevant and actionable insights that directly answer the user query.
        - Ensure insights are concise, factual, and derived from the provided schema/data.
        - Do NOT make assumptions beyond the given schema and sample data.
        - Respond ONLY with a valid JSON object, nothing else.
        - DO NOT include explanations, natural language, markdown, code fences, or comments.
        - DO NOT invent additional keys or change key names.
        - DO NOT add any prefixes like "Here is the JSON".
        - The output must be directly parsable by `json.loads`.

        The JSON object must have exactly this structure:

        {{
          "insights": [
            "string insight 1",
            "string insight 2",
            ...
          ]
        }}

        Example of correct output:

        {{
          "insights": [
            "Freight costs to Germany increased 40% steadily from 1994 to 1995.",
            "Most orders shipped to Germany in 1995 were above $100 freight cost."
          ]
        }}
        """
    return prompt



