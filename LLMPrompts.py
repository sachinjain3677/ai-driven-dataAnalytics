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
        f"- Respond ONLY with one column per line in the format 'column_name: datatype'.\n"
        f"- Do not include explanations, extra text, or JSON formatting.\n"
        f"- Use only these datatypes: int, float, string, datetime.\n"
        f"- For columns that look like numbers but are actually identifiers (e.g., zip codes, order IDs, product codes), use the 'string' datatype.\n"
        f"- If a value is ambiguous, prefer 'string' as a safe default.\n\n"
        f"Row sample: {json.dumps(first_row, indent=2)}"
    )

# Prompt to be given to LLM to generate SQL query
@tracer.chain()
def get_sql_prompt(schema_text: str, samples_text: str, user_query: str) -> str:
    return f"""
    You are an expert DuckDB SQL programmer. Your task is to generate a SQL query from a natural language request.

    Schema of the available tables:
    {schema_text}
    
    Sample data from the tables:
    {samples_text}
    
    Important instructions for DuckDB SQL generation:
    - Use the schema and sample data to understand the table structure and content.
    - For string comparisons that should be case-insensitive, use the `ILIKE` operator instead of `LIKE`.
    - If the user asks for a 'top N' or 'bottom N' result, use `ORDER BY` and `LIMIT N`.
    - When generating queries involving dates, use standard date functions like `YEAR()`, `MONTH()`, `DAY()`, and comparisons with 'YYYY-MM-DD' formatted strings.
    - Do not invent columns or tables. Only use the tables and columns described in the provided schema.
    - Always generate a single, syntactically correct DuckDB SQL query.

    Natural Language User Query:
    "{user_query}"

    Respond **ONLY in this JSON format**, with no additional text or explanations:
    {{
        "sql_query": "SELECT ... FROM ... WHERE ...",
        "explanation": "A brief, user-friendly explanation of what the query does."
    }}
    """


# LLM Prompt to predict graph type according to user query, sample data and dataset schema
@tracer.chain()
def create_graph_prompt(schema_text: str, samples_text: str, user_query: str) -> str:
    prompt = f"""
        You are a data visualization expert. Your task is to determine the most appropriate graph for a user's request based on the data schema and query.

        Schema:
        {schema_text}

        Sample data:
        {samples_text}

        User query: "{user_query}"

        Instructions:
        1.  Analyze the user query to understand the desired analysis (e.g., total, average, distribution).
        2.  Examine the Schema to identify the available categorical and numerical columns.
        3.  Select the best `graph_type` to answer the query. A "Bar" chart is best for comparing a numerical value across categories. A "Histogram" is for viewing the distribution of a single number.
        4.  Select the `x` and `y` columns. `x` is typically the category, and `y` is the number.

        STRICT Response Format:
        - Respond ONLY with a single, valid JSON object.
        - The JSON must contain these exact keys: `graph_type`, `x`, `y`, `title`.
        - `graph_type` must be one of: "Line", "Bar", "Pie", "Scatter", "Histogram".
        - `x` and `y` must be valid column names from the provided Schema.
        - **Crucially, if `graph_type` is "Bar", "Line", "Scatter", or "Pie", the `y` value MUST be a numerical column from the schema. It cannot be null or a categorical column.**
        - `y` can ONLY be `null` if the `graph_type` is "Histogram".
        - `title` should be a descriptive title for the chart.
        - Do not include any explanations, comments, or markdown.

        Example of correct output:

        {{
          "graph_type": "Bar",
          "x": "category_column",
          "y": "numeric_column",
          "title": "Distribution of [Y Column] by [X Column]"
        }}
        """
    return prompt

# LLM Prompt to extract key insights from schema, sample data and user query
@tracer.chain()
def create_insight_prompt(schema_text: str, samples_text: str, user_query: str) -> str:
    prompt = f"""
        You are a senior data analyst reviewing a final dataset that was generated based on a user's query. Your task is to provide concise, intelligent insights about this specific dataset.

        **Crucial Context**: The data you are seeing has **already been filtered** according to the user's original query. You must assume the conditions of the query are true for the data you are analyzing.

        Original User Query: "{user_query}"

        Schema of the Final Dataset:
        {schema_text}

        Sample of the Final Dataset:
        {samples_text}

        Instructions for Generating Insights:
        1.  **Use the User Query as Context**: Your insights should be framed by the user's query. For example, if the query was "show me sales in Germany," you should assume all data is from Germany, even if there is no 'country' column in the final dataset.
        2.  **Analyze the Provided Data**: Focus on finding trends, comparisons, outliers, and key statistics within the given dataset.
        3.  **Be Smart, Not Literal**: Do not state that information is missing if it was part of the original query. Instead, use that context to make your insights more powerful.
        4.  **Be Factual and Concise**: Insights must be derived directly from the data, but interpreted through the lens of the query.

        STRICT Response Format:
        - Respond ONLY with a valid JSON object.
        - The JSON object must contain a single key: `"insights"`.
        - The value of `"insights"` must be a list of strings, where each string is a distinct insight.
        - Do not include explanations, comments, or markdown.

        Example:
        - User Query: "What were the total sales for each product category in 2023?"
        - Data Provided: A table with 'product_category' and 'total_sales' columns.
        - **Correct Insight**: "In 2023, 'Electronics' was the highest-selling category, while 'Home Goods' was the lowest."
        - **Incorrect Insight**: "The data does not specify the year, so I cannot confirm these sales are from 2023."

        Your turn. Generate the insights based on the information provided.

        Example of correct output format:
        {{
          "insights": [
            "Based on the query, [Category A] had the highest average [Metric B].",
            "Within the requested timeframe, there was a significant upward trend in [Metric C]."
          ]
        }}
        """
    return prompt



