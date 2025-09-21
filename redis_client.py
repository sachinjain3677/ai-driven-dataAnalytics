import redis

# --- Central Redis Connection ---
# This client is imported by other modules to ensure a single, shared connection.
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
# ------------------------

def check_redis_connection():
    """Checks the connection to Redis and prints a status message."""
    try:
        redis_client.ping()
        print("Successfully connected to Redis.")
    except redis.exceptions.ConnectionError as e:
        print(f"Could not connect to Redis: {e}")
