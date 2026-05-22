import json
import os
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


DEFAULT_BASE_URL = os.getenv("ASDAGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"


def get_api_key(api_key=None):
    return api_key or os.getenv("ASDAGENT_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY")

def generate_text_embedding_by_llm_api_via_openai(
        messages,
        model_name="text-embedding-ada-002",
        api_key=None,
        base_url=None,
):
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=get_api_key(api_key),
    )
    response = client.embeddings.create(
        model=model_name,
        input=messages
    )
    return response

def generate_text_by_llm_api_via_openai(
        messages,
        model_name,
        base_url,
        api_key=None
):
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url,
        api_key=get_api_key(api_key),
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    return response

def record_api_call(chat_completion, log_path=None):
    """
    Persist call metadata from a ChatCompletion to a JSONL file.
    """
    log_path = Path(log_path or os.getenv("ASDAGENT_USAGE_LOG", "outputs/llm_usage.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    usage_obj = getattr(chat_completion, "usage", None)
    if usage_obj is None:
        usage = None
    elif hasattr(usage_obj, "model_dump"):
        usage = usage_obj.model_dump()
    elif isinstance(usage_obj, dict):
        usage = usage_obj
    else:
        usage = json.loads(usage_obj.json()) if hasattr(usage_obj, "json") else str(usage_obj)

    log_entry = {
        "time": datetime.now().isoformat(),
        "id": getattr(chat_completion, "id", None),
        "model_name": getattr(chat_completion, "model", None),
        "usage": usage,
    }
    with log_path.open("a", encoding="utf-8") as log_file:
        json.dump(log_entry, log_file, ensure_ascii=False)
        log_file.write("\n")

def generate_text_by_llm_api_via_http(
        messages,
        model_name,
        api_key=None,
        base_url=None,
):
    import requests

    url = f"{(base_url or DEFAULT_BASE_URL).rstrip('/')}/chat/completions"
    authorization = "Bearer " + get_api_key(api_key)
    payload = json.dumps({
        "model": model_name,
        "messages": messages
    })
    headers = {
        'Authorization': authorization,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    # print(response.text)
    return response.text

def generate_json_by_llm_api_via_openai_old_version(
        messages,
        model_name,
        base_url=DEFAULT_BASE_URL,
        api_key=None,
):
    from openai import OpenAI

    client = OpenAI(
        api_key=get_api_key(api_key),
        base_url=base_url,
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={"type": "json_object"}
    )
    return response

def generate_json_by_llm_api_via_openai(
        messages,
        model_name,
        json_schema,
        base_url=DEFAULT_BASE_URL,
        api_key=None,
):
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url,
        api_key=get_api_key(api_key),
    )
    response = client.responses.parse(
        model=model_name,
        input=messages,
        text_format=json_schema,
    )
    return response
    # return response.output_parsed

class Step(BaseModel):
    explanation: str
    output: str

class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str

if __name__ == '__main__':
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello!"
        }
    ]
    model_name = os.getenv("ASDAGENT_MODEL", "gpt-4o-mini")
    text = generate_text_by_llm_api_via_openai(
        messages,
        model_name,
        base_url=DEFAULT_BASE_URL,
    )
    record_api_call(
        chat_completion=text
    )
    print(text)
