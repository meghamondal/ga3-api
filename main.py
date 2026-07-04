from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import config
import prompts
from utils import (
    chat,
    embeddings,
    parse_json,
    cosine_similarity,
)

app = FastAPI(
    title="GA3 API",
    version="1.0.0",
)


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

