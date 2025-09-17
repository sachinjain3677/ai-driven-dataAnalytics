# tracing.py
from phoenix.otel import register
import os

PROJECT_NAME = "analytics-app"
PHOENIX_ENDPOINT = "https://app.phoenix.arize.com/s/kirti-bagri81/"

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com/s/kirti-bagri81/"
os.environ["PHOENIX_API_KEY"]="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6MSJ9.RWBfsClU7dXaLBgSzcEhRa3xvaS5TEAIQytW8Us3CPE"

# Initialize Phoenix Tracer Provider globally
tracer_provider = register(
    project_name=PROJECT_NAME,
    endpoint=PHOENIX_ENDPOINT + "v1/traces",
)

# Export the tracer to be used anywhere in the project
tracer = tracer_provider.get_tracer(__name__)
