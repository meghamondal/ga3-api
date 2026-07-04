import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# -----------------------------
# User Configuration
# -----------------------------

EMAIL = os.getenv("EMAIL", "")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")

# -----------------------------
# AIPipe Configuration
# -----------------------------

AIPIPE_BASE = "https://aipipe.org/openai/v1"

# Models
TEXT_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# -----------------------------
# Request Timeout
# -----------------------------

REQUEST_TIMEOUT = 90

print("EMAIL:", EMAIL)
print("TOKEN FOUND:", bool(AIPIPE_TOKEN))