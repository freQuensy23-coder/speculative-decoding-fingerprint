#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import tiktoken


MARKER = "###SPLIT###"
JSON_TASK = (
    'Output the following JSON record EXACTLY 20 times, each on its own line, '
    'no numbering, no code fences, no commentary:\n'
    '{"id":1,"name":"widget","price":9.99,"in_stock":true}'
)
STORY_TASK = "Write a 5-sentence original story about a big mouse."
T_CRIT_95 = {
    2: 12.71,
    3: 4.30,
    4: 3.18,
    5: 2.78,
    6: 2.57,
    7: 2.45,
    8: 2.36,
    9: 2.31,
    10: 2.26,
    11: 2.23,
    12: 2.20,
    13: 2.18,
    14: 2.16,
    15: 2.14,
    16: 2.13,
    17: 2.12,
    18: 2.11,
    19: 2.10,
    20: 2.09,
    25: 2.06,
    30: 2.04,
}


def build_prompt(order):
    split = f'\nOutput a line containing exactly "{MARKER}" and nothing else, then continue.\n'
    if order == "FWD":
        return f"Task 1: {JSON_TASK}\n{split}\nTask 2: {STORY_TASK}\nDo not output anything after the story."
    return f"Task 1: {STORY_TASK}\n{split}\nTask 2: {JSON_TASK}\nDo not output anything after the JSON block."


def request_stream(model, max_tokens, api_key, prompt, nonce):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"nonce:{nonce}"},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    start = time.perf_counter()
    text_parts = []
    char_times = []

    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                now = time.perf_counter() - start
                text_parts.append(content)
                char_times.extend([now] * len(content))

    return "".join(text_parts), char_times


def tokenize_with_times(text, char_times, enc):
    tokens = []
    pos = 0
    for token_id in enc.encode(text, disallowed_special=()):
        piece = enc.decode([token_id])
        end = pos + len(piece)
        t = char_times[min(end - 1, len(char_times) - 1)] if char_times else 0.0
        tokens.append({"t": t, "text": piece})
        pos = end
    return tokens


def split_sections(tokens):
    full = "".join(t["text"] for t in tokens)
    idx = full.find(MARKER)
    if idx < 0:
        return None, None

    pos = 0
    first = None
    last = None
    for i, tok in enumerate(tokens):
        start = pos
        end = pos + len(tok["text"])
        if end > idx and start < idx + len(MARKER):
            first = i if first is None else first
            last = i
        pos = end

    if first is None:
        return None, None
    return tokens[:first], tokens[last + 1 :]


def tokens_per_second(section):
    if len(section) < 3:
        return None
    duration = section[-1]["t"] - section[0]["t"]
    return (len(section) - 1) / max(duration, 1e-9)


def mean_ci(xs):
    n = len(xs)
    if n < 2:
        m = statistics.mean(xs) if xs else float("nan")
        return m, m, m
    m = statistics.mean(xs)
    se = statistics.stdev(xs) / math.sqrt(n)
    tc = T_CRIT_95.get(n - 1, 1.96 if n >= 31 else 2.1)
    return m, m - tc * se, m + tc * se


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-2026-03-05")
    parser.add_argument("--trials-per-order", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--out", default="/tmp/specdec_ordered_raw.json")
    args = parser.parse_args()

    api_key = os.environ["OPENAI_API_KEY"]
    enc = tiktoken.get_encoding("o200k_base")
    schedule = []
    for i in range(args.trials_per_order):
        schedule.extend([("FWD", i), ("REV", i)])
    random.Random(0xBEEF).shuffle(schedule)

    rows = []
    print(
        f"model={args.model}  trials/order={args.trials_per_order}  max_tokens={args.max_tokens}",
        flush=True,
    )

    for order, trial in schedule:
        prompt = build_prompt(order)
        nonce = f"{order}-{trial}-{time.time_ns()}"
        try:
            text, char_times = request_stream(args.model, args.max_tokens, api_key, prompt, nonce)
            tokens = tokenize_with_times(text, char_times, enc)
            section_1, section_2 = split_sections(tokens)
        except Exception as exc:
            print(f"  [{order} {trial}] ERROR: {exc}", file=sys.stderr, flush=True)
            continue

        if section_1 is None:
            print(f"  [{order} {trial}] marker not found", file=sys.stderr, flush=True)
            continue

        if order == "FWD":
            json_section, story_section = section_1, section_2
        else:
            story_section, json_section = section_1, section_2

        json_tps = tokens_per_second(json_section)
        story_tps = tokens_per_second(story_section)
        if json_tps is None or story_tps is None:
            print(f"  [{order} {trial}] section too short", file=sys.stderr, flush=True)
            continue

        rows.append(
            {
                "order": order,
                "trial": trial,
                "json_tps": json_tps,
                "story_tps": story_tps,
                "json_tokens": len(json_section),
                "story_tokens": len(story_section),
            }
        )
        print(
            f"  {order} {trial:02d}  JSON: {len(json_section):3d}tok {json_tps:5.1f} tok/s  |  "
            f"STORY: {len(story_section):3d}tok {story_tps:5.1f} tok/s  |  "
            f"ratio J/S={json_tps / story_tps:.2f}",
            flush=True,
        )

    if not rows:
        print("no valid rows")
        return

    print()
    for order in ("FWD", "REV"):
        subset = [r for r in rows if r["order"] == order]
        if not subset:
            continue
        json_tps = [r["json_tps"] for r in subset]
        story_tps = [r["story_tps"] for r in subset]
        ratios = [j / s for j, s in zip(json_tps, story_tps)]
        mj, lj, hj = mean_ci(json_tps)
        ms, ls, hs = mean_ci(story_tps)
        mr, lr, hr = mean_ci(ratios)
        first = "JSON" if order == "FWD" else "STORY"
        print(f"=== ORDER={order}  (first section = {first})  n={len(subset)} ===")
        print(f"  JSON   tok/s  mean={mj:6.2f}  95% CI [{lj:6.2f}, {hj:6.2f}]")
        print(f"  STORY  tok/s  mean={ms:6.2f}  95% CI [{ls:6.2f}, {hs:6.2f}]")
        print(f"  ratio  JSON/STORY  mean={mr:5.3f}  95% CI [{lr:5.3f}, {hr:5.3f}]")
        print(f"  trials with JSON faster than STORY: {sum(r > 1.0 for r in ratios)}/{len(ratios)}")
        print()

    fwd = [r["json_tps"] / r["story_tps"] for r in rows if r["order"] == "FWD"]
    rev = [r["json_tps"] / r["story_tps"] for r in rows if r["order"] == "REV"]
    mf, lf, hf = mean_ci(fwd)
    mr, lr, hr = mean_ci(rev)
    print("=== content vs position verdict ===")
    print(f"  FWD (JSON first)  ratio J/S = {mf:.3f} [{lf:.3f}, {hf:.3f}]")
    print(f"  REV (STORY first) ratio J/S = {mr:.3f} [{lr:.3f}, {hr:.3f}]")
    if lf > 1.0 and lr > 1.0:
        print("  -> J/S > 1 in BOTH orders. Result is CONTENT-driven, not position.")
    elif hf < 1.0 and hr < 1.0:
        print("  -> J/S < 1 in BOTH orders. Result is content-driven in the opposite direction.")
    else:
        print("  -> inconclusive.")

    Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
