# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv

# Instrumentation in Arize Phoenix
from phoenix.otel import register

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com/s/kirti-bagri81/"
os.environ["PHOENIX_API_KEY"]="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6MSJ9.RWBfsClU7dXaLBgSzcEhRa3xvaS5TEAIQytW8Us3CPE"

def load_env():
    _ = load_dotenv(find_dotenv(), override=True)

def get_phoenix_endpoint():
    load_env()
    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    return phoenix_endpoint

PROJECT_NAME = "analytics-app"
tracer_provider = register(
    project_name=PROJECT_NAME,
    endpoint= get_phoenix_endpoint() + "v1/traces"
)

tracer = tracer_provider.get_tracer(__name__)