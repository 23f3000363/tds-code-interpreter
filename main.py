from fastapi import FastAPI
from pydantic import BaseModel
import io
import contextlib
import traceback
import os
import json
import re
from openai import OpenAI

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


def execute_python_code(code: str) -> dict:
    output_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code)

        return {
            "success": True,
            "output": output_buffer.getvalue()
        }

    except Exception:
        return {
            "success": False,
            "output": traceback.format_exc()
        }


def analyze_error_with_ai(code: str, error_traceback: str):
    client = OpenAI(
        api_key=os.environ.get("GEMINI_API_KEY"),
        base_url="https://aipipe.org/openai/v1"
    )

    prompt = f"""
Analyze this Python code and its error traceback.

Identify the exact line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{error_traceback}

Return ONLY a JSON object in this exact format:
{{"error_lines": [3]}}

Do not include any other text.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You identify exact Python error line numbers."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    text = response.choices[0].message.content.strip()

    try:
        result = json.loads(text)
        return result["error_lines"]

    except Exception:
        numbers = re.findall(r"\b\d+\b", text)
        return [int(n) for n in numbers]


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):

    result = execute_python_code(request.code)

    if result["success"]:
        return {
            "error": [],
            "result": result["output"]
        }

    error_lines = analyze_error_with_ai(
        request.code,
        result["output"]
    )

    return {
        "error": error_lines,
        "result": result["output"]
    }