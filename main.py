from typing import Any, Dict
import json
import httpx

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import base64
from statistics import mean, median, pstdev, pvariance, mode

import config
import prompts

from audio import router


from utils import (
    chat,
    embeddings,
    parse_json,
    cosine_similarity,
)

from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="GA3 API",
    version="1.0.0",
)

app.include_router(router)

'''
# ----------------------------------------------------
# Q6 Debug
# ----------------------------------------------------

last_debug_info = {}
'''
# ----------------------------------------------------
# CORS
# ----------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ----------------------------------------------------
# Request Models
# ----------------------------------------------------

class InvoiceRequest(BaseModel):
    invoice_text: str

# ----------------------------------------------------
# Root Endpoint
# ----------------------------------------------------

@app.get("/")
async def root():

    return {
        "status": "running",
        "email": config.EMAIL
    }

# ----------------------------------------------------
# Q2
# Image Question Answering
# ----------------------------------------------------

@app.post("/answer-image")
async def answer_image(request: Request):

    body = await request.json()

    image_base64 = body.get("image_base64", "")
    question = body.get("question", "")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":
                        prompts.IMAGE_QA_PROMPT
                        + "\n\nQuestion:\n"
                        + question,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url":
                        f"data:image/png;base64,{image_base64}"
                    },
                },
            ],
        }
    ]

    try:

        response = await chat(
            messages=messages,
            model=config.VISION_MODEL,
            max_tokens=300,
        )

        output = parse_json(response)

        answer = output.get("answer", "")

    except Exception:

        answer = ""

    return {
        "answer": str(answer)
    }

# ----------------------------------------------------
# Q3
# Invoice Extraction
# ----------------------------------------------------

@app.post("/extract")
async def extract(request: Request):
    body = await request.json()

    # ---------- Q3 ----------
    if "invoice_text" in body:
        text = body.get("invoice_text", "")

        prompt = f"""
        You are an invoice extraction engine.

        Return ONLY valid JSON.

        Use EXACTLY these keys:

        {{
          "invoice_no": string | null,
          "date": string | null,
          "vendor": string | null,
          "amount": number | null,
          "tax": number | null,
          "currency": string | null
        }}

        Rules:
        - invoice_no = invoice number.
        - date MUST be ISO format YYYY-MM-DD.
        - amount = subtotal BEFORE tax.
        - Never return the grand total.
        - tax = tax amount only.
        - currency must be the ISO currency code.
          Examples:
          Rs., ₹ -> INR
          $ -> USD
          € -> EUR
          £ -> GBP
          ¥ -> JPY
        - amount and tax MUST be JSON numbers.
        - Never include currency symbols.
        - Use null if a value is missing.

        Invoice:

        {text}
        """

        try:
            response = await chat(
                [{"role": "user", "content": prompt}],
                model=config.TEXT_MODEL,
                max_tokens=600
            )

            out = parse_json(response)

        except Exception as e:
            print("Q3 ERROR:", repr(e))

            out = {
                "invoice_no": None,
                "date": None,
                "vendor": None,
                "amount": None,
                "tax": None,
                "currency": None,
            }

        keys = [
            "invoice_no",
            "date",
            "vendor",
            "amount",
            "tax",
            "currency"
        ]

        return {k: out.get(k) for k in keys}

    # ---------- Q7 ----------
    text = body.get("text", "")
    schema = body.get("schema", {})

    prompt = f"""
    You are an expert information extraction engine.

    Your task is to extract structured information from the invoice/document.

    Return ONLY valid JSON.

    IMPORTANT:
    - Return JSON ONLY.
    - Do NOT explain.
    - Do NOT return the schema.
    - Do NOT wrap JSON inside markdown.
    - Use the provided schema ONLY as the output contract.

    The output MUST match the schema exactly.

    Extraction Rules:

    - vendor:
      Exact company/person issuing the invoice.

    - currency:
      ISO 4217 currency code.
      Examples:
      ₹ or Rs. -> INR
      $ -> USD
      € -> EUR
      £ -> GBP
      ¥ -> JPY

    - total_amount:
      Integer only.
      Ignore commas.
      Ignore currency symbols.

    - invoice_date:
      Convert to YYYY-MM-DD.

    - due_in_days:
      Net 30 -> 30
      Two weeks -> 14
      Pay within 45 days -> 45

    - is_paid:
      true if paid.
      false if unpaid.

    - priority:
      low
      normal
      high
      urgent

    - contact_email:
      Lowercase.

    - line_items:
      Preserve order.
      Each item has:
      sku
      quantity
      unit_price

    - item_count:
      Number of objects in line_items.

    Use null if a value cannot be extracted.

    JSON Schema:

    {json.dumps(schema, indent=2)}

    Document:

    {text}
    """

    try:
        response = await chat(
            [{"role": "user", "content": prompt}],
            model=config.TEXT_MODEL,
            max_tokens=1500
        )

        out = parse_json(response)

    except Exception as e:
        print("Q7 ERROR:", repr(e))
        out = {}

    return out

from pydantic import BaseModel
from typing import Dict, Any

class Q3Request(BaseModel):
    invoice_text: str


@app.post("/extract-q3")
async def extract_q3(body: Q3Request):

    prompt = f"""
You are an invoice extraction engine.

Return ONLY valid JSON.

Use EXACTLY these keys:

{{
  "invoice_no": string | null,
  "date": string | null,
  "vendor": string | null,
  "amount": number | null,
  "tax": number | null,
  "currency": string | null
}}

Rules:
- invoice_no = invoice number.
- date MUST be ISO format YYYY-MM-DD.
- amount = subtotal BEFORE tax.
- Never return the grand total.
- tax = tax amount only.
- currency must be the ISO currency code.
  Examples:
  Rs., ₹ -> INR
  $ -> USD
  € -> EUR
  £ -> GBP
  ¥ -> JPY
- amount and tax MUST be JSON numbers.
- Never include currency symbols.
- Use null if a value is missing.

Invoice:

{body.invoice_text}
"""

    try:

        response = await chat(
            [{"role": "user", "content": prompt}],
            model=config.TEXT_MODEL,
            max_tokens=600,
        )

        out = parse_json(response)

    except Exception as e:

        return {
            "error": str(e)
        }

    return out


# Q7 from pydantic import BaseModel
from typing import Dict, Any

class Q7Request(BaseModel):
    text: str
    schema: Dict[str, Any]


@app.post("/extract-q7")
async def extract_q7(body: Q7Request):
    text = body.text
    schema = body.schema
    prompt = f"""
    You are an expert information extraction engine.

    Your task is to extract structured information from the invoice/document.

    Return ONLY valid JSON.

    IMPORTANT:
    - Return JSON ONLY.
    - Do NOT explain.
    - Do NOT return the schema.
    - Do NOT wrap JSON inside markdown.
    - Use the provided schema ONLY as the output contract.

    The output MUST match the schema exactly.

    Extraction Rules:

    - vendor:
      Exact company/person issuing the invoice.

    - currency:
      ISO 4217 currency code.
      Examples:
      ₹ or Rs. -> INR
      $ -> USD
      € -> EUR
      £ -> GBP
      ¥ -> JPY

    - total_amount:
      Integer only.
      Ignore commas.
      Ignore currency symbols.
      Example:
      "$12,480" -> 12480

    - invoice_date:
      Convert to YYYY-MM-DD.

    - due_in_days:
      Examples:
      Net 30 -> 30
      Due in two weeks -> 14
      Pay within 45 days -> 45

    - is_paid:
      true if invoice says Paid / Paid in Full.
      false if Awaiting Payment / Unpaid / Outstanding.

    - priority:
      One of:
      low
      normal
      high
      urgent

    - contact_email:
      Lowercase.

    - line_items:
      Preserve order.
      Every item has:
      sku
      quantity
      unit_price

    - item_count:
      Number of objects inside line_items.

    Use null if a value cannot be extracted.

    JSON Schema:

    {json.dumps(schema, indent=2)}

    Document:

    {text}
    """

    try:

        response = await chat(
            [{"role": "user", "content": prompt}],
            model=config.TEXT_MODEL,
            max_tokens=1200,
        )

        out = parse_json(response)

    except Exception as e:
        print("Q7 ERROR:", repr(e))
        out = {}

    return out

# ----------------------------------------------------
# Q4 Helper
# ----------------------------------------------------

from dateutil.parser import parse


def coerce(value, typ):

    if value is None:
        return None

    try:

        t = typ.lower()

        if t in ["integer", "int"]:
            return int(float(value))

        if t in ["float", "number"]:
            return float(value)

        if t == "boolean":

            if isinstance(value, bool):
                return value

            return str(value).lower() in [
                "true",
                "1",
                "yes",
            ]

        if t == "date":

            return parse(str(value)).date().isoformat()

        return str(value)

    except Exception:

        return None
    
# ----------------------------------------------------
# Q4
# Dynamic Schema Extraction
# ----------------------------------------------------

@app.post("/dynamic-extract")
async def dynamic_extract(request: Request):

    body = await request.json()

    text = body.get("text", "")
    schema = body.get("schema", {})

    keys = list(schema.keys())

    prompt = (
        "Extract information from the text.\n\n"
        "Return ONLY valid JSON.\n"
        "Return EXACTLY the keys provided in the schema.\n"
        "No extra keys.\n"
        "No missing keys.\n"
        "The JSON MUST contain EXACTLY these keys:\n\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Rules:\n"
        "- Dates -> YYYY-MM-DD\n"
        "- integer -> JSON integer\n"
        "- float/number -> JSON number\n"
        "- boolean -> true/false\n"
        "- string -> plain text\n"
        "- Use null if a value cannot be found.\n\n"
        f"TEXT:\n{text}"
    )

    try:

        response = await chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=config.TEXT_MODEL,
            max_tokens=800,
        )

        out = parse_json(response)

    except Exception:

        out = {}

    return {
        k: coerce(out.get(k), schema[k])
        for k in keys
    }


class RankRequest(BaseModel):
    query: str
    candidates: list[str]


# ----------------------------------------------------
# Q8
# Semantic Search Ranking
# ----------------------------------------------------

@app.post("/rank")
async def rank(body: RankRequest):

    texts = [body.query] + body.candidates

    vectors = await embeddings(texts)

    query_vector = vectors[0]
    candidate_vectors = vectors[1:]

    scores = []

    for index, vector in enumerate(candidate_vectors):

        score = cosine_similarity(query_vector, vector)

        scores.append((index, score))

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    ranking = [idx for idx, _ in scores[:3]]

    return {
        "ranking": ranking
    }



class SolveRequest(BaseModel):
    problem: str

# ----------------------------------------------------
# Q9
# Arithmetic Word Problem Solver
# ----------------------------------------------------

@app.post("/solve")
async def solve(body: SolveRequest):

    prompt = f"""
You are an expert mathematical reasoning assistant.

Solve the following arithmetic word problem carefully.

Instructions:

- Ignore irrelevant or distracting numbers.
- Perform the calculations step by step.
- Double-check the arithmetic.
- Return ONLY valid JSON.
- Do not use markdown.
- The answer must be a JSON number (not a string).

Return EXACTLY this format:

{{
  "reasoning": "...",
  "answer": 0
}}

Problem:

{body.problem}
"""

    try:

        response = await chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=config.TEXT_MODEL,
            max_tokens=1000,
        )

        out = parse_json(response)

        answer = out.get("answer")

        try:
            answer = int(float(answer))
        except Exception:
            answer = 0

        reasoning = str(out.get("reasoning", ""))

    except Exception:

        reasoning = ""
        answer = 0

    return {
        "reasoning": reasoning,
        "answer": answer,
    }

'''

######################################## Q6 ###########################################################################

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

@app.get("/debug")
async def debug():
    return last_debug_info

# ----------------------------------------------------
# Q6
# Audio Statistics
# ----------------------------------------------------

@app.post("/answer-audio")
async def answer_audio(body: AudioRequest):

    global last_debug_info

    last_debug_info = {
        "audio_id": body.audio_id
    }

    audio_b64 = body.audio_base64
    transcript = ""

    try:

        audio = base64.b64decode(audio_b64)

        async with httpx.AsyncClient(
            timeout=120
        ) as client:

            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text":
                                "Transcribe this Korean audio accurately. "
                                "Return ONLY the Korean transcript."
                            },
                            {
                                "inlineData": {
                                    "mimeType": "audio/mp3",
                                    "data": audio_b64,
                                }
                            },
                        ]
                    }
                ]
            }

            response = await client.post(
                "https://aipipe.org/geminiv1beta/models/gemini-2.5-flash-lite:generateContent",
                headers={
                    "Authorization":
                    f"Bearer {config.AIPIPE_TOKEN}"
                },
                json=payload,
            )

            response.raise_for_status()

            data = response.json()

            transcript = (
                data["candidates"][0]
                ["content"]["parts"][0]["text"]
                .strip()
            )

    except Exception as e:

        last_debug_info["error"] = str(e)

        transcript = ""

    last_debug_info["transcript"] = transcript

    prompt = (
        "The following Korean transcript describes a dataset.\n"
        "Extract the dataset and requested statistics.\n\n"
        "Return ONLY valid JSON in this format:\n"
        "{\n"
        '  "columns": [],\n'
        '  "data_rows": [],\n'
        '  "requested_stats": [],\n'
        '  "explicit_stats": {}\n'
        "}\n\n"
        "Allowed statistic names:\n"
        "mean, std, variance, min, max, median, mode, "
        "range, allowed_values, value_range, correlation\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )

    columns = []
    data_rows = []
    requested_stats = []
    explicit_stats = {}

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

        last_debug_info["parsed"] = parsed

    except Exception as e:

        last_debug_info["parse_error"] = str(e)

    return {
        "rows": 0,
        "columns": [],
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
        "correlation": []
    }

    '''