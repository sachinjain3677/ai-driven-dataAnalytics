# ai-driven-dataAnalytics
A data analytics tool tapping into the capabilities of AI for query generation, data loading and relevant workbook creation

# How to setup 

## 1. Update brew
brew update
## 2. Install Python, Node, Git
brew install python@3.11 node git
## 3. create virtualenv
python3 -m venv venv
source venv/bin/activate
## 4. install ollama and run mistral
curl -fsSL https://ollama.ai/install.sh | sh
ollama run mistral
## 5. Install python packages
pip install duckdb
pip install uvicorn
pip install fastapi
pip install sqlglot
pip install requests
pip install langfuse
pip install plotly
## 6. Run the fastAPI app 
uvicorn app:app --reload
## 7. In a separate terminal, run the curl command
curl -X POST "http://127.0.0.1:8000/generate_sql" -H "Content-Type: application/json" -d '{"query": "Give me the freight details of all the orders shipped to Germany in the year 1995"}'