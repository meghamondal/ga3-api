import base64
import httpx
import time
import pandas as pd

from fastapi import APIRouter
from pydantic import BaseModel

import config
from utils import (
    chat,
    parse_json,
)

router = APIRouter()

# ----------------------------------------------------
# Request Model
# ----------------------------------------------------

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


# ----------------------------------------------------
# Debug
# ----------------------------------------------------

last_debug_info = {}

start = time.perf_counter()

@router.get("/debug")
async def debug():
    return last_debug_info


# ----------------------------------------------------
# Q6: Answer Audio Endpoint
# ----------------------------------------------------

@router.post("/answer-audio")
async def answer_audio(body: AudioRequest):
    global last_debug_info

    last_debug_info = {
        "audio_id": body.audio_id
    }

    audio_b64 = body.audio_base64
    transcript = ""

    # 1. Transcribe the audio using Gemini
    try:
        audio = base64.b64decode(audio_b64)
        mime = "audio/mp3"

        if audio.startswith(b"RIFF"):
            mime = "audio/wav"
        elif audio.startswith(b"OggS"):
            mime = "audio/ogg"

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Transcribe this Korean audio accurately. "
                                "Return ONLY the Korean transcript."
                            )
                        },
                        {
                            "inlineData": {
                                "mimeType": mime,
                                "data": audio_b64,
                            }
                        },
                    ]
                }
            ]
        }

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://aipipe.org/geminiv1beta/models/gemini-2.5-flash-lite:generateContent",
                headers={
                    "Authorization": f"Bearer {config.AIPIPE_TOKEN}"
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            transcript = data["candidates"][0]["content"]["parts"][0]["text"].strip()

    except Exception as e:
        last_debug_info["error"] = str(e)
        transcript = ""

    last_debug_info["transcript"] = transcript

    last_debug_info["gemini_seconds"] = round(
        time.perf_counter() - start,
        2
    )

    gpt_start = time.perf_counter()

    # 2. Extract structured JSON using GPT-4o
    prompt = (
        "The transcript (Korean) describes a tabular dataset and statistics.\n"
        "Extract structured information.\n\n"
        "Return ONLY JSON with this structure:\n"
        "{\n"
        '  "columns": [],\n'
        '  "data_rows": [],\n'
        '  "num_rows": null,\n'
        '  "requested_stats": [],\n'
        '  "explicit_stats": {}\n'
        "}\n\n"
        "Allowed requested_stats values:\n"
        "mean,std,variance,min,max,median,mode,range,"
        "allowed_values,value_range,correlation\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )

    columns = []
    data_rows = []
    requested_stats = []
    explicit_stats = {}
    num_rows = None

    try:
        response = await chat(
            [{"role": "user", "content": prompt}],
            model="gpt-4o",
            max_tokens=1500,
        )

        parsed = parse_json(response)

        columns = parsed.get("columns", [])
        data_rows = parsed.get("data_rows", [])
        requested_stats = parsed.get("requested_stats", [])
        explicit_stats = parsed.get("explicit_stats", {})
        num_rows = parsed.get("num_rows")

        # Recover column names from explicit_stats if the LLM omitted them
        if not columns and explicit_stats:
            columns = []

            for value in explicit_stats.values():
                if isinstance(value, dict):
                    for key in value.keys():
                        if key not in columns:
                            columns.append(key)

        last_debug_info["parsed"] = parsed

        last_debug_info["gpt_seconds"] = round(
            time.perf_counter() - gpt_start,
            2
            )
        last_debug_info["total_seconds"] = round(
            time.perf_counter() - start,
            2
            )

    except Exception as e:
        last_debug_info["parse_error"] = str(e)

    # Fallback to all stats if the LLM didn't specify
    if not requested_stats:
        requested_stats = [
            "mean", "std", "variance", "min", "max",
            "median", "mode", "range", "allowed_values",
            "value_range", "correlation",
        ]
        
    rows = num_rows if num_rows is not None else len(data_rows)

    # Standardize result structure
    result = {
        "rows": rows,
        "columns": columns,
        "mean": {},
        "std": {},
        "variance": {},
        "min": {},
        "max": {},
        "median": {},
        "mode": {},
        "range": {},
        "allowed_values": {},
        "value_range": {},
        "correlation": [],
    }

    # 3. Compute statistics safely using Pandas
    try:
        # Load the extracted data into a DataFrame
        if data_rows:
            df = pd.DataFrame(data_rows)

            if columns and len(columns) == len(df.columns):
                df.columns = columns
        else:
            df = pd.DataFrame()
        
        # If columns weren't perfectly matched but exist, apply them safely
        if columns and len(df.columns) == len(columns):
            df.columns = columns

        # Ensure numeric columns are actually floats for calculation
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        for col in df.columns:
            col_data = df[col].dropna()
            
            if col_data.empty:
                continue

            if "mean" in requested_stats:
                result["mean"][col] = col_data.mean()

            if "std" in requested_stats:
                # Pandas std() handles sample size naturally
                result["std"][col] = col_data.std() if len(col_data) > 1 else 0.0

            if "variance" in requested_stats:
                result["variance"][col] = col_data.var() if len(col_data) > 1 else 0.0

            if "min" in requested_stats:
                result["min"][col] = float(col_data.min())

            if "max" in requested_stats:
                result["max"][col] = float(col_data.max())

            if "median" in requested_stats:
                result["median"][col] = col_data.median()

            if "mode" in requested_stats:
                modes = col_data.mode()
                result["mode"][col] = float(modes.iloc[0]) if not modes.empty else float(col_data.iloc[0])

            if "range" in requested_stats:
                result["range"][col] = float(col_data.max() - col_data.min())

            if "value_range" in requested_stats:
                result["value_range"][col] = [float(col_data.min()), float(col_data.max())]

            if "allowed_values" in requested_stats:
                result["allowed_values"][col] = sorted(list(set(col_data.tolist())))

        # Compute Correlation Matrix
        if "correlation" in requested_stats and len(df.columns) > 1:
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                # Calculate correlation, fill NaNs with 0, and convert to list of lists
                corr_matrix = numeric_df.corr().fillna(0).values.tolist()
                result["correlation"] = corr_matrix

    except Exception as e:
        last_debug_info["pandas_error"] = str(e)

    # 4. Update with explicit stats that were stated verbatim in the audio transcript
    for key, value in explicit_stats.items():
        # Case 1:
        # {"median":{"income":45000}}
        if key in result and isinstance(result[key], dict):
            result[key].update(value)

        # Case 2:
        # {"성별":{"allowed_values":["남성","여성"]}}
        elif isinstance(value, dict):

            for stat_name, stat_value in value.items():

                if stat_name == "allowed_values":
                    result["allowed_values"][key] = stat_value

                elif stat_name == "value_range":
                    result["value_range"][key] = stat_value

                elif stat_name in result and isinstance(result[stat_name], dict):
                    result[stat_name][key] = stat_value

        elif key == "correlation" and isinstance(value, list):
            result["correlation"] = value

    return result