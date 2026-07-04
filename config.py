import os

# -----------------------------
# User Configuration
# -----------------------------

# Your IITM email (set in Render environment variables)
EMAIL = os.getenv("EMAIL", "")

# Your AIPipe API Token (set in Render environment variables)
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