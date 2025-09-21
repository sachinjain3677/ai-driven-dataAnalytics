# ai-driven-dataAnalytics
A data analytics tool tapping into the capabilities of AI for query generation, data loading and relevant workbook creation

# How to setup 

## 1. Update brew
brew update
brew install ffmpeg
## 2. Install Python, Node, Git
brew install python@3.11 node git
## 3. create virtualenv
python3 -m venv venv
source venv/bin/activate
## 4. installing required models
* curl -fsSL https://ollama.ai/install.sh | sh
* ollama run mistral
* wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
* unzip vosk-model-small-en-us-0.15.zip
## 5. Install python packages
pip install duckdb
pip install uvicorn
pip install fastapi
pip install sqlglot
pip install requests
pip install langfuse
pip install plotly
pip install python-multipart
pip install vosk
pip install transformers
pip install pandas
pip install kaleido
pip install arize-phoenix-otel
pip install google.generativeai
## 6. Run the fastAPI app 
uvicorn app:app --reload
## 7. In a separate terminal, run the curl command
curl -X POST "http://127.0.0.1:8000/generate_sql" -H "Content-Type: application/json" -d '{"query": "Give me total freight sent to each country in the year 1995"}'
curl -X POST "http://localhost:8000/generate_sql_from_audio" -F "audio=@sample.wav"