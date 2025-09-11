# Prompt to be given to LLM to generate SQL query
def get_sql_prompt(schema_text: str, samples_text: str, user_query: str) -> str:
    return f"""
    You are a SQL expert. Based on the schema below:
    {schema_text}
    
    Sample Data:
    {samples_text}
    
    Important instructions for DuckDB:
    - The `orderdate` column stores timestamps as strings in the format MM/DD/YYYY HH:MM:SS AM/PM.
    - DuckDB requires date functions (YEAR(), MONTH(), comparisons) to be applied on DATE or TIMESTAMP types.
    - Always convert `orderdate` safely using:
        CAST(STRPTIME(SUBSTR(orderdate,1,10), '%m/%d/%Y') AS DATE)
      before using it in any function or comparison.
    - Use standard DATE literals in the format 'YYYY-MM-DD' instead of 'MM/DD/YYYY HH:MM:SS AM/PM'.
    - Do not assume `00:00:00 AM` is valid — use only the date portion when filtering.
    
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
