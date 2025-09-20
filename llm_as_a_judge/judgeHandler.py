import os
import json
# from openai import OpenAI
from opentelemetry import trace
from tracing import tracer
import google.generativeai as genai
import re

# # OpenAI client for judge
# openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Configure Gemini client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---- Prompt Builders ---- #
def get_sql_judge_prompt(input_text: str, output: str) -> str:
    return f"""
        You are an expert SQL reviewer.
        
        Your task is to judge whether the generated SQL query correctly answers the user's request given the provided input (which may include schema/context).
        
        Consider:
        - Does the SQL logic match the user's request?  
        - Are the tables and columns used correctly?  
        - Are filters, aggregations, and joins applied as intended?  
        - Would this query return the expected result set?  
        
        Input (user request + schema + sample data representing the tables in DB):
        {input_text}
        
        Generated SQL:
        {output}
        
        Respond **ONLY in this JSON format**:
        {{
          "verdict": "correct" | "incorrect",
          "explanation": "Brief explanation of why the SQL is correct or not."
        }}
    """

def get_dtype_judge_prompt(input_text: str, output: str) -> str:
    return f"""
        You are a strict evaluator for CSV column datatype inference.

        Model Input:
        {input_text}
        
        Model Output:
        {output}
        
        Evaluation criteria:
        1. Does the output provide exactly one datatype for each column in the input row?
        2. Are the datatypes restricted to: int, float, string, datetime?
        3. Is the mapping reasonable given the sample values?  
           - Numeric values without decimals → int  
           - Numeric values with decimals → float  
           - Textual or mixed values → string  
           - Date/time formats → datetime  
           - POSTALCODE must always be string
        4. No extra text, explanations, or JSON structures are allowed. Only col: dtype lines.
        
        Respond ONLY in JSON:
        {{
            "verdict": "correct" | "incorrect",
          "explanation": "Brief reason why the output is correct or what issues exist"
        }}
    """

def get_graph_judge_prompt(input_text: str, output: str) -> str:
    return f"""You are a strict evaluator for graph metadata generation in a data visualization workflow.

        The input provided to the smaller model was:
        {input_text}
        
        The output produced by the smaller model was:
        {output}
        
        Evaluation Criteria:
        1. Graph Type:
           - Must be one of: "Line", "Bar", "Pie", "Scatter", "Histogram".
           - If the user query explicitly requests a graph type, it must match that type.
        
        2. X and Y Columns:
           - Must match actual columns from the schema included in the input_text.
           - For "Histogram", only "x" is required and "y" must be null.
           - For "Pie", "x" should be categorical and "y" numeric.
           - For "Line", "Bar", "Scatter", both x and y must be valid schema columns and logically aligned (time vs value, category vs value, etc.).
           - No invented or null columns (unless allowed by graph type).
        
        3. Title:
           - Must reflect the user query intent.
           - Should reference the chosen columns and be concise.
        
        4. JSON Format:
           - Must be valid JSON with exactly the keys: "graph_type", "x", "y", "title".
           - No extra keys, comments, markdown, or explanations.
        
        Your task:  
        Determine whether the output_text is *correct* or *incorrect* for the given input_text.
        
        Respond ONLY in JSON format:
        
        {{
            "verdict": "correct" | "incorrect",
          "explanation": "Brief reason why it is correct or what issues exist"
        }}
    """

def get_analysis_judge_prompt(input_text: str, output: str) -> str:
    return f"""You are judging analysis generation.
Input request: {input_text}
Generated analysis: {output}
Return JSON with 'verdict' and 'explanation'."""


PROMPT_BUILDERS = {
    "sql": get_sql_judge_prompt,
    "dtypes": get_dtype_judge_prompt,
    "graph": get_graph_judge_prompt,
    "analysis": get_analysis_judge_prompt,
}


def get_response_text(response) -> str:
    """
    Extracts text from Gemini response safely.
    Handles cases where finish_reason != 1 (STOP).
    """
    if not response.candidates:
        return ""

    candidate = response.candidates[0]

    # finish_reason 1 = STOP (normal), 2 = SAFETY, 3 = LENGTH, 4 = OTHER
    if candidate.finish_reason != 1:
        return f"[no output: finish_reason={candidate.finish_reason}]"

    if candidate.content.parts:
        return "".join(part.text for part in candidate.content.parts if hasattr(part, "text"))
    return ""

# -----------------------
# NEW: LLM Judge function
# -----------------------
@tracer.chain(name="llm.judge_response")
def judge_response_with_gemini(task_type: str, input_text: str, output: str) -> dict:
    """
    Calls Gemini to judge correctness for different task types.
    task_type ∈ {"sql", "dtypes", "graph", "analysis"}
    """
    if task_type not in PROMPT_BUILDERS:
        raise ValueError(f"Invalid task_type '{task_type}'. Must be one of {list(PROMPT_BUILDERS.keys())}")

    judge_prompt = PROMPT_BUILDERS[task_type](input_text, output)

    model = genai.GenerativeModel("gemini-2.5-flash")
    print("JUDGE_PROMPT: ", judge_prompt)
    response = model.generate_content(
        [
            "Respond ONLY with a JSON object containing two keys: `verdict` and `explanation`. "
            "Do not add extra text, only return JSON.",
            judge_prompt,
        ],
        generation_config={"temperature": 0, "max_output_tokens": 2048},
    )

    print("\n\nGEMINI RESPONSE, response: ", response)
    raw_msg = get_response_text(response).strip()
    print("\n\nGEMINI RESPONSE, RAW_MSG: ", raw_msg)
    cleaned = clean_json_response(raw_msg)
    print("\n\nGEMINI RESPONSE, cleaned: ", cleaned)

    try:
        result = json.loads(cleaned)
    except Exception:
        result = {"verdict": "error", "explanation": cleaned}

    # Attach attributes for Phoenix tracing
    current_span = trace.get_current_span()
    current_span.set_attribute("judge.task_type", task_type)
    current_span.set_attribute("judge.input", input_text)
    current_span.set_attribute("judge.output", output)
    current_span.set_attribute("judge.verdict", result.get("verdict"))
    current_span.set_attribute("judge.explanation", result.get("explanation"))

    return result


def clean_json_response(raw: str) -> str:
    """
    Cleans Gemini output to extract valid JSON.
    Removes markdown fences like ```json ... ```
    """
    raw = raw.strip()

    # Remove ```json ... ``` or ``` ... ```
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)   # remove opening fence
        raw = re.sub(r"\n?```$", "", raw)            # remove closing fence

    return raw.strip()

def get_sql_judge_prompt(sql_prompt: str, generated_sql: str) -> str:
    return f"""
        You are an expert SQL reviewer.
        
        Your task is to judge whether the generated SQL query correctly answers the user's request given the provided input (which may include schema/context).
        
        Consider:
        - Does the SQL logic match the user's request?  
        - Are the tables and columns used correctly?  
        - Are filters, aggregations, and joins applied as intended?  
        - Would this query return the expected result set?  
        
        Input (user request + schema + sample data representing the tables in DB):
        {sql_prompt}
        
        Generated SQL:
        {generated_sql}
        
        Respond **ONLY in this JSON format**:
        {{
          "verdict": "correct" | "incorrect",
          "explanation": "Brief explanation of why the SQL is correct or not."
        }}
    """