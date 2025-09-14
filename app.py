from fastapi import FastAPI, Request, File, UploadFile
from io import BytesIO
import subprocess
import json
import wave
from vosk import Model, KaldiRecognizer

from dataLoad import connect_to_duckdb, load_data_into_duckdb_with_llm, get_schemas, get_top_rows
from LLMPrompts import *
from SqlResponseHandler import *
from GraphGenerator import *
from vectordb_handler import VectorDBHandler

app = FastAPI()

vosk_model_path = "./vosk-model-small-en-us-0.15"
model = Model(vosk_model_path)

schema_text = None
samples_text = None

@app.on_event("startup")
async def startup_event():
    global schema_text, samples_text
    # Code here runs once, when the server starts
    print("Server is starting...")

    # Connect to DuckDB
    connect_to_duckdb()

    #Load Data into DuckDB
    load_data_into_duckdb_with_llm()

    # Define schema_text and samples_text (hardcoded for now)
    schema_text = get_schemas()
    samples_text = get_top_rows()

@app.post("/generate_sql")
async def generate_sql(request: Request):
    body = await request.json()
    user_query = body.get("query")  # Natural language query
    process_user_query(user_query)


# Converts any audio file to wav which can be further processed to text by speech-to-text llm
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

        # Step 5: Process transcription
        process_user_query(transcription)
        print("[INFO] Transcription sent to process_user_query")

        return {"transcription": transcription.strip()}

    except Exception as e:
        print(f"[EXCEPTION] Error in generate_sql_from_audio: {type(e).__name__}: {e}")
        return {"error": str(e)}

# Processing of received user query to fetch data and plot graph
def process_user_query(user_query: str):
    global schema_text, samples_text
    print("\nBuilding SQL generation prompt...")
    sql_prompt = get_sql_prompt(schema_text, samples_text, user_query)

    print("\nGetting SQL query from LLM...")
    generated_sql = get_sql_query_from_llm(sql_prompt)

    print("\nGenerated SQL:\n", generated_sql)

    validated_sql = validate_and_normalize_sql(generated_sql)

    print("\nExecuting validated SQL...")
    query_results = execute_sql(validated_sql)
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

    # Store query result in vectorDB
    handler = VectorDBHandler(db_path="./chroma_orders_db")

    handler.insert_dataframe(
        df=query_results,
        collection_name="query_results_collection",
        table_name="query_results"
    )

    results = handler.query("query_results_collection", user_query, n_results=50)

    # Initialize an empty string
    results_str = ""

    # Append formatted metadata for each result
    for meta in results["metadatas"][0]:
        formatted_meta = json.dumps(meta, indent=2)  # pretty JSON format
        results_str += f"{formatted_meta}\n\n"

    analysis_query = "Which country ordered the most freight based on this data"
    analysis_prompt = create_insight_prompt(query_results_schema_text, results_str, analysis_query)

    analysis_response = call_llm(analysis_prompt, span_name="ollama_generate_sql", external_id="request_12345")
    print("Result Insights: ", analysis_response)

    handler.clear_collection("query_results_collection")