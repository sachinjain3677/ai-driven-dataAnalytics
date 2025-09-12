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
        raise RuntimeError(f"ffmpeg error: {process.stderr.decode()}")
    return BytesIO(process.stdout)

#####################################
# Speech to text API
#####################################
@app.post("/generate_sql_from_audio")
async def generate_sql_from_audio(audio: UploadFile = File(...)):
    try:
        # Step 1: Read uploaded audio
        audio_bytes = await audio.read()
        wav_io = convert_to_wav(audio_bytes)

        # Step 2: Load WAV for Vosk
        wav_io.seek(0)
        wf = wave.open(wav_io, "rb")

        # Step 3: Initialize recognizer
        rec = KaldiRecognizer(model, wf.getframerate())

        # Step 4: Process audio frames
        transcription = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                transcription += res.get("text", "") + " "

        # Final result
        res = json.loads(rec.FinalResult())
        transcription += res.get("text", "")

        # Step 5: Process transcription
        print("\nAudio to text:", transcription)
        process_user_query(transcription)

        return {"transcription": transcription.strip()}

    except Exception as e:
        return {"error": str(e)}

# Proceessing of received user query to fetch data and plot graph
def process_user_query(user_query: str):
    print("\nBuilding SQL generation prompt...")
    sql_prompt = get_sql_prompt(schema_text, samples_text, user_query)

    print("\nGetting SQL query from LLM...")
    generated_sql = get_sql_query_from_llm(sql_prompt)

    print("\nGenerated SQL:\n", generated_sql)

    validated_sql = validate_and_normalize_sql(generated_sql)

    print("\nExecuting validated SQL...")
    query_results = execute_sql(validated_sql)

    print("\nBuilding graph metadata prompt...")
    graph_prompt = create_graph_prompt(", ".join(query_results.columns), query_results.head(3).to_string(index=False), user_query)

    print("\nGetting graph metadata from LLM...")
    metadata = get_graph_metadata_from_llm(graph_prompt)

    print("\nPlotting graph...")
    fig = plot_graph(query_results, metadata)

