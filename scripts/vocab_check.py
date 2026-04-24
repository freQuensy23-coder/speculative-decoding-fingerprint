#!/usr/bin/env python3
"""Figure out which tiktoken encoding matches gpt-5.4 by comparing
counted tokens against API-reported usage.completion_tokens."""
import json, os, urllib.request, tiktoken
from pathlib import Path

KEY = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-5.4-2026-03-05"

def call(prompt, max_tok=200):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tok}
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

prompts = [
    'Output this JSON once, nothing else: {"id":1,"name":"widget","price":9.99,"in_stock":true}',
    'Write a single sentence about a big mouse exploring a vast library at midnight.',
    'Output exactly 10 random 20-character hex strings, one per line, nothing else.',
]

for p in prompts:
    resp = call(p, max_tok=300)
    text = resp["choices"][0]["message"]["content"]
    api_ct = resp["usage"]["completion_tokens"]
    line = f"\nprompt: {p[:60]}...\nAPI usage.completion_tokens = {api_ct}\n"
    for enc_name in ["o200k_base", "o200k_harmony", "cl100k_base"]:
        enc = tiktoken.get_encoding(enc_name)
        n = len(enc.encode(text, disallowed_special=()))
        line += f"  {enc_name:15s}: {n}  (diff {n-api_ct:+d})\n"
    print(line)
