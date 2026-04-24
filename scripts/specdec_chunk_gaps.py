#!/usr/bin/env python3
"""
Speculative-decoding fingerprint test — token-level analysis.

For each streamed response we record the arrival time and text of EVERY
SSE chunk. From that we derive:
  (1) aggregate tokens/sec (for sanity)
  (2) inter-chunk gap distribution (ms between consecutive chunks)
  (3) bytes-per-chunk distribution
  (4) burst detection: count of consecutive chunks arriving within <1ms
      of each other (fingerprint of multi-token batch-accept)

Hypothesis: under speculative decoding,
  - on PREDICTABLE text the draft hits ~always, so we see bursts of
    several tokens arriving within microseconds, then short pauses.
  - on RANDOM text the draft misses, so we see uniform token-by-token
    arrivals with ~constant gap.

Condition PREDICTABLE: repeat fixed JSON 40 times.
Condition RANDOM:     40 lines of random hex.
"""

import json
import os
import sys
import time
import urllib.request
import statistics
from pathlib import Path

MODEL = "gpt-5.4-2026-03-05"      # full gpt-5.4 (pinned)
TRIALS = 15                       # per condition (gpt-5.4 is pricier)
MAX_TOKENS = 600
KEY = os.environ["OPENAI_API_KEY"]

PREDICTABLE_PROMPT = (
    "Output the following JSON record EXACTLY 40 times, each on its own line, "
    "no other text, no numbering, no code fences, no surrounding commentary:\n"
    '{"id":1,"name":"widget","price":9.99,"in_stock":true}'
)

RANDOM_PROMPT = (
    "Output 40 lines. Each line must be a random 20-character string of "
    "lowercase hex digits (0-9, a-f). No other text, no numbering, no code fences. "
    "Make each line as random and unpredictable as you can."
)

def stream_once(prompt: str, nonce: str):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": f"nonce:{nonce}"},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    t_start = time.perf_counter()
    chunks = []   # list of (t_rel_seconds, content_str)
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
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                chunks.append((time.perf_counter() - t_start, content))
    return chunks, usage

# ---------- stats helpers ----------
def q(xs, p):
    xs = sorted(xs); n = len(xs)
    if n == 0: return float("nan")
    k = (n - 1) * p; f = int(k); c = min(f + 1, n - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def hist(xs, edges):
    counts = [0] * (len(edges) - 1)
    for x in xs:
        for i in range(len(edges) - 1):
            if edges[i] <= x < edges[i+1]:
                counts[i] += 1; break
        else:
            counts[-1] += 1
    return counts

def analyze(label, runs):
    print(f"\n=== {label} (n={len(runs)}) ===")
    if not runs: return None

    per_run_tps = []
    per_run_ttft = []
    per_run_ntoks = []
    all_gaps_ms = []
    all_chunk_sizes = []
    burst_chunks_total = 0
    total_chunks = 0

    for r in runs:
        chunks = r["chunks"]
        usage = r["usage"] or {}
        if len(chunks) < 2: continue
        t_first = chunks[0][0]
        t_last  = chunks[-1][0]
        per_run_ttft.append(t_first)
        gen_s = t_last - t_first
        n_tok = (usage.get("completion_tokens") or len(chunks))
        per_run_ntoks.append(n_tok)
        per_run_tps.append((n_tok - 1) / max(gen_s, 1e-9))

        for i in range(1, len(chunks)):
            gap_ms = (chunks[i][0] - chunks[i-1][0]) * 1000.0
            all_gaps_ms.append(gap_ms)
            if gap_ms < 1.0:
                burst_chunks_total += 1
            total_chunks += 1
        for (_, c) in chunks:
            all_chunk_sizes.append(len(c))

    print(f"  completion_tokens  median={statistics.median(per_run_ntoks):.0f}  "
          f"range=[{min(per_run_ntoks)}, {max(per_run_ntoks)}]")
    print(f"  TTFT (s)           median={statistics.median(per_run_ttft):.3f}  "
          f"IQR=[{q(per_run_ttft,.25):.3f}, {q(per_run_ttft,.75):.3f}]")
    print(f"  tok/s aggregate    median={statistics.median(per_run_tps):.2f}  "
          f"IQR=[{q(per_run_tps,.25):.2f}, {q(per_run_tps,.75):.2f}]")

    # gap distribution
    print(f"  inter-chunk gap ms  n={len(all_gaps_ms)}")
    print(f"    p10={q(all_gaps_ms,.10):.2f}  p25={q(all_gaps_ms,.25):.2f}  "
          f"p50={q(all_gaps_ms,.50):.2f}  p75={q(all_gaps_ms,.75):.2f}  p90={q(all_gaps_ms,.90):.2f}")
    edges = [0, 0.1, 1, 5, 10, 20, 40, 80, 160, 10_000]
    counts = hist(all_gaps_ms, edges)
    for i in range(len(edges)-1):
        pct = 100.0 * counts[i] / len(all_gaps_ms)
        bar = "#" * int(pct/2)
        print(f"    [{edges[i]:>6.1f},{edges[i+1]:>7.1f}) ms  {counts[i]:>6}  {pct:5.1f}%  {bar}")
    print(f"  bursts (<1ms gap)   {burst_chunks_total}/{total_chunks} = "
          f"{100*burst_chunks_total/max(total_chunks,1):.1f}%")

    # chunk size distribution
    print(f"  bytes/chunk         median={statistics.median(all_chunk_sizes):.1f}  "
          f"p90={q(all_chunk_sizes,.90):.1f}  max={max(all_chunk_sizes)}")

    return {
        "median_tps": statistics.median(per_run_tps),
        "burst_pct": 100*burst_chunks_total/max(total_chunks,1),
        "gap_p50": q(all_gaps_ms, .50),
        "gap_p90": q(all_gaps_ms, .90),
    }

def main():
    print(f"model={MODEL}  trials/cond={TRIALS}  max_tokens={MAX_TOKENS}")
    pred_runs, rand_runs = [], []
    for i in range(TRIALS):
        for label, prompt, bucket in [
            ("PREDICTABLE", PREDICTABLE_PROMPT, pred_runs),
            ("RANDOM",      RANDOM_PROMPT,      rand_runs),
        ]:
            nonce = f"{label}-{i}-{time.time_ns()}"
            try:
                chunks, usage = stream_once(prompt, nonce)
            except Exception as e:
                print(f"  [{i} {label}] ERROR: {e}", file=sys.stderr); continue
            if len(chunks) < 2:
                print(f"  [{i} {label}] empty stream", file=sys.stderr); continue
            bucket.append({"chunks": chunks, "usage": usage})
            n = (usage or {}).get("completion_tokens", len(chunks))
            gen_s = chunks[-1][0] - chunks[0][0]
            print(f"  {i:02d} {label:11s}  chunks={len(chunks):4d}  toks={n:4d}  "
                  f"ttft={chunks[0][0]:.3f}s  gen={gen_s:.3f}s  tok/s={(n-1)/max(gen_s,1e-6):.1f}")

    s_pred = analyze("PREDICTABLE (fixed JSON x40)", pred_runs)
    s_rand = analyze("RANDOM (40 hex lines)", rand_runs)

    if s_pred and s_rand:
        print("\n=== comparison ===")
        print(f"  tok/s ratio pred/rand      : {s_pred['median_tps']/s_rand['median_tps']:.2f}x")
        print(f"  burst% pred vs rand        : {s_pred['burst_pct']:.1f}%  vs  {s_rand['burst_pct']:.1f}%")
        print(f"  median gap pred vs rand    : {s_pred['gap_p50']:.2f}ms  vs  {s_rand['gap_p50']:.2f}ms")
        print(f"  p90 gap pred vs rand       : {s_pred['gap_p90']:.2f}ms  vs  {s_rand['gap_p90']:.2f}ms")

    Path("/tmp/specdec_raw.json").write_text(json.dumps(
        {"model": MODEL, "predictable": pred_runs, "random": rand_runs}, indent=2))
    print("\nraw saved to /tmp/specdec_raw.json")

if __name__ == "__main__":
    main()
