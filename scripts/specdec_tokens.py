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


MARKER = "###STORY###"
PROMPT = (
    'Task 1: Output the following JSON record EXACTLY 20 times, each on its own line, '
    'no numbering, no code fences, no commentary:\n'
    '{"id":1,"name":"widget","price":9.99,"in_stock":true}\n\n'
    f'Task 2: After the last JSON line, output a line containing exactly "{MARKER}" '
    'and nothing else. Then write a 5-sentence original story about a big mouse. '
    'Do not output anything after the story.'
)
T_CRIT_95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
    11: 2.228,
    12: 2.201,
    13: 2.179,
    14: 2.160,
    15: 2.145,
    16: 2.131,
    17: 2.120,
    18: 2.110,
    19: 2.101,
    20: 2.093,
    25: 2.060,
    30: 2.042,
}


def request_stream(model, max_tokens, api_key, nonce):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"nonce:{nonce}"},
            {"role": "user", "content": PROMPT},
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
    usage = None

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
            if obj.get("usage"):
                usage = obj["usage"]
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

    return "".join(text_parts), char_times, usage


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
    if len(section) < 2:
        return None
    duration = section[-1]["t"] - section[0]["t"]
    return (len(section) - 1) / max(duration, 1e-9)


def gaps_ms(section):
    return [
        (section[i]["t"] - section[i - 1]["t"]) * 1000.0
        for i in range(1, len(section))
    ]


def mean_ci(xs):
    n = len(xs)
    if n < 2:
        m = statistics.mean(xs) if xs else float("nan")
        return m, m, m
    m = statistics.mean(xs)
    se = statistics.stdev(xs) / math.sqrt(n)
    tc = T_CRIT_95.get(n - 1, 1.96 if n >= 31 else 2.1)
    return m, m - tc * se, m + tc * se


def quantile(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def bootstrap_median_ci(xs, rounds=2000):
    if not xs:
        return float("nan"), float("nan"), float("nan")
    rnd = random.Random(0xC0FFEE)
    if len(xs) > 5000:
        xs = [xs[rnd.randrange(len(xs))] for _ in range(5000)]
    meds = []
    for _ in range(rounds):
        sample = [xs[rnd.randrange(len(xs))] for _ in range(len(xs))]
        meds.append(statistics.median(sample))
    meds.sort()
    return statistics.median(xs), meds[int(rounds * 0.025)], meds[int(rounds * 0.975) - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-2026-03-05")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--out", default="/tmp/specdec_tokens_raw.json")
    args = parser.parse_args()

    api_key = os.environ["OPENAI_API_KEY"]
    enc = tiktoken.get_encoding("o200k_base")
    rows = []
    all_gaps_a = []
    all_gaps_b = []

    print(f"model={args.model}  trials={args.trials}  max_tokens={args.max_tokens}", flush=True)

    for i in range(args.trials):
        nonce = f"tokens-{i}-{time.time_ns()}"
        try:
            text, char_times, usage = request_stream(args.model, args.max_tokens, api_key, nonce)
            tokens = tokenize_with_times(text, char_times, enc)
            section_a, section_b = split_sections(tokens)
        except Exception as exc:
            print(f"  [{i}] ERROR: {exc}", file=sys.stderr, flush=True)
            continue

        if section_a is None:
            print(f"  [{i}] marker not found", file=sys.stderr, flush=True)
            continue

        tps_a = tokens_per_second(section_a)
        tps_b = tokens_per_second(section_b)
        if tps_a is None or tps_b is None:
            print(f"  [{i}] section too short", file=sys.stderr, flush=True)
            continue

        dur_a = section_a[-1]["t"] - section_a[0]["t"]
        dur_b = section_b[-1]["t"] - section_b[0]["t"]
        gap_a = gaps_ms(section_a)
        gap_b = gaps_ms(section_b)
        all_gaps_a.extend(gap_a)
        all_gaps_b.extend(gap_b)

        rows.append(
            {
                "trial": i,
                "json_tps": tps_a,
                "story_tps": tps_b,
                "json_tokens": len(section_a),
                "story_tokens": len(section_b),
                "json_duration": dur_a,
                "story_duration": dur_b,
                "usage": usage,
            }
        )

        print(
            f"  {i:02d}  A: {len(section_a):3d}tok / {dur_a:5.2f}s = {tps_a:5.1f} tok/s "
            f"(gap p50={quantile(gap_a, .5):5.1f} p90={quantile(gap_a, .9):6.1f} ms)  ||  "
            f"B: {len(section_b):3d}tok / {dur_b:5.2f}s = {tps_b:5.1f} tok/s "
            f"(gap p50={quantile(gap_b, .5):5.1f} p90={quantile(gap_b, .9):6.1f} ms)",
            flush=True,
        )

    if not rows:
        print("no valid trials")
        return

    a = [r["json_tps"] for r in rows]
    b = [r["story_tps"] for r in rows]
    ratios = [x / y for x, y in zip(a, b)]
    diffs = [x - y for x, y in zip(a, b)]

    m_a, l_a, h_a = mean_ci(a)
    m_b, l_b, h_b = mean_ci(b)
    m_r, l_r, h_r = mean_ci(ratios)
    m_d, l_d, h_d = mean_ci(diffs)

    print(f"\n=== per-section tokens/sec (n={len(rows)}, 95% CI Student-t) ===")
    print(f"  A predictable : mean={m_a:6.2f}  95% CI [{l_a:6.2f}, {h_a:6.2f}]")
    print(f"  B creative    : mean={m_b:6.2f}  95% CI [{l_b:6.2f}, {h_b:6.2f}]")
    print(f"  ratio A/B     : mean={m_r:6.3f}  95% CI [{l_r:6.3f}, {h_r:6.3f}]")
    print(f"  trials with A faster than B: {sum(r > 1.0 for r in ratios)}/{len(ratios)}")

    med_a, lg_a, hg_a = bootstrap_median_ci(all_gaps_a)
    med_b, lg_b, hg_b = bootstrap_median_ci(all_gaps_b)
    print("\n=== pooled inter-token gap median (ms), 95% bootstrap CI ===")
    print(f"  A predictable : median={med_a:6.3f}  CI [{lg_a:6.3f}, {hg_a:6.3f}]  (n={len(all_gaps_a)})")
    print(f"  B creative    : median={med_b:6.3f}  CI [{lg_b:6.3f}, {hg_b:6.3f}]  (n={len(all_gaps_b)})")

    print("\n=== gap percentiles (ms) ===")
    for p in (50, 75, 90, 95, 99):
        print(f"  p{p:<2}  A={quantile(all_gaps_a, p / 100):7.2f}   B={quantile(all_gaps_b, p / 100):7.2f}")

    print("\n=== paired A-B tok/s difference ===")
    print(f"  mean diff = {m_d:6.2f}  95% CI [{l_d:6.2f}, {h_d:6.2f}]  tok/s")

    Path(args.out).write_text(json.dumps({"model": args.model, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
