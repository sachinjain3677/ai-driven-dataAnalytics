from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from io import BytesIO
import subprocess
import json
import wave
import os
import shutil
import base64
from vosk import Model, KaldiRecognizer

from dataLoad import connect_to_duckdb, load_data_into_duckdb_with_llm, get_schemas, get_top_rows, reset_database, reset_dtype_cache
from LLMPrompts import *
from SqlResponseHandler import *
from GraphGenerator import *
from vectordb_handler import VectorDBHandler
from tracing import tracer
from LLMResponseGenerator import call_llm_analysis_generation

from llm_as_a_judge.judgeHandler import judge_response_with_gemini
from redis_client import redis_client, check_redis_connection
from datetime import datetime
import pandas as pd

llm_as_a_judge = os.getenv("LLM_AS_A_JUDGE", "false").lower() in ("true", "1", "yes")

app = FastAPI()

# --- Custom JSON Serializer ---
def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

vosk_model_path = "./vosk-model-small-en-us-0.15"
model = Model(vosk_model_path)

@app.on_event("startup")
async def startup_event():
    print("Server is starting...")
    # Connect to DuckDB
    connect_to_duckdb()
    # Check Redis connection
    check_redis_connection()

@app.post("/upload_csv")
@tracer.tool()
async def upload_csv(file: UploadFile = File(...)):
    upload_folder = "uploads"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file_path = os.path.join(upload_folder, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    table_name = os.path.splitext(file.filename)[0]
    load_data_into_duckdb_with_llm(file_path, table_name)

    # Get schema and samples
    schema_text = get_schemas()
    samples_text = get_top_rows()

    # Store in Redis, using the custom serializer for datetime objects
    redis_client.set("schema_text", json.dumps(schema_text, default=json_serial))
    redis_client.set("samples_text", json.dumps(samples_text, default=json_serial))

    return {"message": f"File '{file.filename}' uploaded and processed successfully."}

@app.post("/reset")
@tracer.tool()
async def reset_state():
    """Resets the application state by clearing the database and uploaded files."""
    print("Resetting application state...")

    # 1. Clear the DuckDB database
    reset_database()

    # 2. Clear the in-memory dtype cache
    reset_dtype_cache()

    # 3. Clear the state in Redis
    redis_client.delete("schema_text", "samples_text")
    print("State cleared from Redis.")

    # 4. Clear the uploads directory
    upload_folder = "uploads"
    if os.path.exists(upload_folder):
        shutil.rmtree(upload_folder)
        print(f"'{upload_folder}' directory has been removed.")
    os.makedirs(upload_folder)
    print(f"'{upload_folder}' directory has been recreated.")

    print("Application state has been successfully reset.")
    return {"message": "Application state has been successfully reset."}

@app.post("/generate_sql")
@tracer.tool()
async def generate_sql(request: Request):
    body = await request.json()
    user_query = body.get("query")  # Natural language query
    graph_path, insights = process_user_query(user_query)
    print("\nTranscription sent to process_user_query")

    # Encode the image to Base64
    with open(graph_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    return {
        "graph": encoded_image,
        "insights": insights
    }


# Converts any audio file to wav which can be further processed to text by speech-to-text llm
@tracer.chain()
def convert_to_wav(audio_bytes: bytes) -> BytesIO:
    """Convert any audio file to mono 16kHz WAV in-memory using ffmpeg."""
    print(f"[INFO] Starting audio conversion, input size: {len(audio_bytes)} bytes")

    process = subprocess.run(
        [
            "ffmpeg",
            "-i", "pipe:0",
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            "pipe:1"
        ],
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if process.returncode != 0:
        err_msg = process.stderr.decode(errors="ignore")
        print(f"[ERROR] ffmpeg failed with return code {process.returncode}")
        print(f"[ERROR] ffmpeg stderr: {err_msg}")
        raise RuntimeError(f"ffmpeg error: {err_msg}")

    print(f"[INFO] Audio conversion successful, output size: {len(process.stdout)} bytes")
    return BytesIO(process.stdout)


#####################################
# Speech to text API
#####################################
@app.post("/generate_sql_from_audio")
@tracer.tool()
async def generate_sql_from_audio(audio: UploadFile = File(...)):
    try:
        print(f"[INFO] Received audio file: {audio.filename}")

        # Step 1: Read uploaded audio
        audio_bytes = await audio.read()
        print(f"[INFO] Audio file size: {len(audio_bytes)} bytes")

        wav_io = convert_to_wav(audio_bytes)
        print("[INFO] Converted audio to WAV format")

        # Step 2: Load WAV for Vosk
        wav_io.seek(0)
        wf = wave.open(wav_io, "rb")
        print(f"[INFO] WAV file opened, frame rate: {wf.getframerate()}, channels: {wf.getnchannels()}")

        # Step 3: Initialize recognizer
        rec = KaldiRecognizer(model, wf.getframerate())
        print("[INFO] KaldiRecognizer initialized")

        # Step 4: Process audio frames
        transcription = ""
        frame_count = 0
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            frame_count += 1
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                text_piece = res.get("text", "")
                transcription += text_piece + " "
                print(f"[DEBUG] Frame {frame_count}: Partial transcription -> '{text_piece}'")

        # Final result
        res = json.loads(rec.FinalResult())
        transcription += res.get("text", "")
        print(f"[INFO] Final transcription: '{transcription.strip()}'")

        # Step 5: Process transcription and get results
        graph_path, insights = process_user_query(transcription)
        print("[INFO] Transcription sent to process_user_query")

        # Encode the image to Base64
        with open(graph_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        return {
            "transcription": transcription.strip(),
            "graph": encoded_image,
            "insights": insights
        }

    except Exception as e:
        print(f"[EXCEPTION] Error in generate_sql_from_audio: {type(e).__name__}: {e}")
        return {"error": str(e)}

# Processing of received user query to fetch data and plot graph
def process_user_query(user_query: str) -> tuple[str, str]:
    print("\nBuilding SQL generation prompt...")
    
    # Retrieve data from Redis
    schema_text_json = redis_client.get("schema_text")
    samples_text_json = redis_client.get("samples_text")

    if not schema_text_json or not samples_text_json:
        # This is a fallback, ideally the client should handle this state
        raise HTTPException(status_code=400, detail="No data has been uploaded. Please upload a CSV first.")

    schema_text = json.loads(schema_text_json)
    samples_text = json.loads(samples_text_json)

    with tracer.start_as_current_span(
            "execute_sql_query",
            openinference_span_kind="chain"
    ) as span:
        span.set_input(value=user_query)
        sql_prompt = get_sql_prompt(schema_text, samples_text, user_query)

        print("\nGetting SQL query from LLM...")
        generated_sql = get_sql_query_from_llm(sql_prompt)

        print("\nGenerated SQL:\n", generated_sql)

        validated_sql = validate_and_normalize_sql(generated_sql)

        print("\nExecuting validated SQL...")
        query_results = execute_sql(validated_sql)
        span.set_output(value=query_results)

    with tracer.start_as_current_span(
            "Generate_graph_from_query",
            openinference_span_kind="chain"
    ) as span1:
        span1.set_input(value=query_results)
        query_results_schema_text = ", ".join(query_results.columns)
        query_results_samples_text = query_results.head(3).to_string(index=False)
    
        print("\nBuilding graph metadata prompt...")
        graph_prompt = create_graph_prompt(query_results_schema_text, query_results_samples_text, user_query)
    
        print("\nGetting graph metadata from LLM...")
        metadata = get_graph_metadata_from_llm(graph_prompt)

        print("\nPlotting graph...")
        fig = plot_graph(query_results, metadata)
    
        # Save the generated graph image locally
        graph_image_path = "generated_graph.png"
        fig.write_image(graph_image_path)
        span1.set_output(value=metadata)

    # Store query result in vectorDB
    handler = VectorDBHandler(db_path="./chroma_orders_db")

    handler.insert_dataframe(
        df=query_results,
        collection_name="query_results_collection",
        table_name="query_results"
    )

    with tracer.start_as_current_span(
            "Get_Insights_from_Query",
            openinference_span_kind="chain"
    ) as span2:
        results = handler.query("query_results_collection", user_query, n_results=50)
        span2.set_output(value=results)
    
        # Initialize an empty string
        results_str = ""
    
        # Append formatted metadata for each result
        for meta in results["metadatas"][0]:
            formatted_meta = json.dumps(meta, indent=2)  # pretty JSON format
            results_str += f"{formatted_meta}\n\n"
    
        analysis_prompt = create_insight_prompt(query_results_schema_text, results_str, user_query)
    
        analysis_response = call_llm_analysis_generation(analysis_prompt)

        if (llm_as_a_judge):
            judge_response_with_gemini("analysis", analysis_prompt, analysis_response)

        print("Result Insights: ", analysis_response)
        span2.set_output(value=analysis_response)
    
        handler.clear_collection("query_results_collection")
    
        return(graph_image_path, analysis_response)