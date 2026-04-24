#!/usr/bin/env python3
"""
Token-level within-stream speculative-decoding test.

Per-chunk we capture:
  - arrival timestamp
  - streamed delta text
  - post-hoc tokenization with tiktoken over the final output

One prompt per trial, producing two sections in a single stream:
  A: repeat fixed JSON 20x  (predictable)
  B: 5-sentence mouse story (creative)
split by the "###STORY###" marker (marker tokens themselves excluded).

Per-trial metric: generation-phase tokens/sec in each section.
Reported: mean ± 95% CI (Student-t) across trials, plus bootstrap CI
on median inter-token gap.
"""

import json, os, sys, time, math, statistics, random, urllib.request
from pathlib import Path
import tiktoken

MODEL = "gpt-5.4-2026-03-05"
TRIALS = 20
MAX_TOKENS = 900
KEY = os.environ["OPENAI_API_KEY"]
MARKER = "###STORY###"

PROMPT = (
    'Task 1: Output the following JSON record EXACTLY 20 times, each on its own line, '
    'no numbering, no code fences, no commentary:\n'
    '{"id":1,"name":"widget","price":9.99,"in_stock":true}\n\n'
    'Task 2: After the last JSON line, output a line containing exactly "###STORY###" '
    'and nothing else. Then write a 5-sentence original story about a big mouse. '
    'Do not output anything after the story.'
)

# tiktoken fallback
try:
    ENC = tiktoken.encoding_for_model("gpt-4o")  # o200k_base; same family
except Exception:
    ENC = tiktoken.get_encoding("o200k_base")

def stream_once(nonce):
    """Stream the response; return list of (t, char) pairs plus the
    final text. Each character inherits the arrival time of its SSE chunk.
    Tokenization happens post-hoc with tiktoken."""
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": f"nonce:{nonce}"},
            {"role": "user", "content": PROMPT},
        ],
        "max_completion_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json",
                 "Accept": "text/event-stream"},
        method="POST",
    )
    t_start = time.perf_counter()
    char_times = []   # per-byte arrival time
    text_parts = []
    usage = None
    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            try: obj = json.loads(payload)
            except json.JSONDecodeError: continue
            if obj.get("usage"): usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices: continue
            now = time.perf_counter() - t_start
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                text_parts.append(content)
                for ch in content:
                    char_times.append(now)
    full = "".join(text_parts)
    # tokenize the full output
    ids = ENC.encode(full, disallowed_special=())
    tokens = []
    pos = 0
    for tid in ids:
        piece = ENC.decode([tid])
        # each token's arrival time = arrival time of its LAST character
        # (the chunk in which the token was finalized)
        end = pos + len(piece)
        if end > len(char_times):
            # shouldn't happen but guard
            t = char_times[-1] if char_times else 0.0
        else:
            t = char_times[end-1]
        tokens.append({"t": t, "tok": piece})
        pos = end
    return tokens, usage, False

def split_at_marker(tokens):
    full = "".join(t["tok"] for t in tokens)
    idx = full.find(MARKER)
    if idx < 0: return None, None
    # walk tokens to find indices spanning MARKER
    acc = 0; first = None; last = None
    for i, t in enumerate(tokens):
        start = acc
        end = acc + len(t["tok"])
        # token overlaps marker?
        if end > idx and start < idx + len(MARKER):
            if first is None: first = i
            last = i
        acc = end
    if first is None: return None, None
    A = tokens[:first]
    B = tokens[last+1:]
    return A, B

def tps(sec_tokens):
    if len(sec_tokens) < 2: return None
    dur = sec_tokens[-1]["t"] - sec_tokens[0]["t"]
    return (len(sec_tokens) - 1) / max(dur, 1e-9)

def gaps_ms(sec_tokens):
    return [(sec_tokens[i]["t"] - sec_tokens[i-1]["t"]) * 1000.0
            for i in range(1, len(sec_tokens))]

# --- stats helpers ---
def mean_ci_t(xs, conf=0.95):
    n = len(xs)
    if n < 2: return (statistics.mean(xs) if xs else float("nan"), 0.0, 0.0)
    m = statistics.mean(xs); sd = statistics.stdev(xs); se = sd / math.sqrt(n)
    # two-sided t critical; approximate via normal for small n or use table
    # use exact via scipy-free approximation: t ≈ z for n>=30, else fetch table
    t_table = {2:12.706, 3:4.303, 4:3.182, 5:2.776, 6:2.571, 7:2.447, 8:2.365,
               9:2.306, 10:2.262, 11:2.228, 12:2.201, 13:2.179, 14:2.160,
               15:2.145, 16:2.131, 17:2.120, 18:2.110, 19:2.101, 20:2.093,
               25:2.060, 30:2.042}
    df = n - 1
    tc = t_table.get(df, 1.96 if df >= 30 else 2.1)
    return m, m - tc*se, m + tc*se

def bootstrap_median_ci(xs, B=2000, conf=0.95):
    if not xs: return (float("nan"), float("nan"), float("nan"))
    rnd = random.Random(0xC0FFEE)
    n = len(xs)
    meds = []
    for _ in range(B):
        sample = [xs[rnd.randrange(n)] for _ in range(n)]
        meds.append(statistics.median(sample))
    meds.sort()
    a = (1 - conf) / 2
    return statistics.median(xs), meds[int(B*a)], meds[int(B*(1-a))-1]

def q(xs, p):
    xs = sorted(xs); n = len(xs)
    if n == 0: return float("nan")
    k = (n-1)*p; f = int(k); c = min(f+1, n-1)
    return xs[f] + (xs[c]-xs[f])*(k-f)

def main():
    print(f"model={MODEL}  trials={TRIALS}  max_tokens={MAX_TOKENS}", flush=True)
    per_trial = []    # (tpsA, tpsB, nA, nB, durA, durB)
    all_gapsA, all_gapsB = [], []
    lp_used = 0

    for i in range(TRIALS):
        nonce = f"TOK-{i}-{time.time_ns()}"
        try:
            tokens, usage, lp_seen = stream_once(nonce)
        except Exception as e:
            print(f"  [{i}] ERROR: {e}", file=sys.stderr, flush=True); continue
        if lp_seen: lp_used += 1
        A, B = split_at_marker(tokens)
        if A is None:
            print(f"  [{i}] marker not found", file=sys.stderr, flush=True); continue
        tA, tB = tps(A), tps(B)
        if tA is None or tB is None:
            print(f"  [{i}] section too short A={len(A)} B={len(B)}", file=sys.stderr, flush=True)
            continue
        durA = A[-1]["t"] - A[0]["t"]
        durB = B[-1]["t"] - B[0]["t"]
        per_trial.append((tA, tB, len(A), len(B), durA, durB))
        gA, gB = gaps_ms(A), gaps_ms(B)
        all_gapsA.extend(gA); all_gapsB.extend(gB)
        print(f"  {i:02d}  A: {len(A):3d}tok / {durA:5.2f}s = {tA:5.1f} tok/s "
              f"(gap p50={q(gA,.5):5.1f} p90={q(gA,.9):6.1f} ms)  ||  "
              f"B: {len(B):3d}tok / {durB:5.2f}s = {tB:5.1f} tok/s "
              f"(gap p50={q(gB,.5):5.1f} p90={q(gB,.9):6.1f} ms)", flush=True)

    if not per_trial:
        print("no valid trials"); return

    tpsA = [x[0] for x in per_trial]
    tpsB = [x[1] for x in per_trial]
    ratios = [a/b for a,b in zip(tpsA, tpsB)]

    mA, lA, hA = mean_ci_t(tpsA)
    mB, lB, hB = mean_ci_t(tpsB)
    mR, lR, hR = mean_ci_t(ratios)

    print(f"\nlogprobs used in {lp_used}/{len(per_trial)} trials (rest used tiktoken fallback)")
    print(f"\n=== per-section tokens/sec (n={len(per_trial)}, 95% CI Student-t) ===")
    print(f"  A predictable : mean={mA:6.2f}  95% CI [{lA:6.2f}, {hA:6.2f}]")
    print(f"  B creative    : mean={mB:6.2f}  95% CI [{lB:6.2f}, {hB:6.2f}]")
    print(f"  ratio A/B     : mean={mR:6.3f}  95% CI [{lR:6.3f}, {hR:6.3f}]")

    # paired test: all ratios > 1?
    n_above = sum(1 for r in ratios if r > 1.0)
    print(f"  trials with A faster than B: {n_above}/{len(ratios)}")

    # bootstrap medians on pooled gaps
    print(f"\n=== pooled inter-token gap median (ms), 95% bootstrap CI ===")
    def short(xs):
        # limit bootstrap sample size for speed
        if len(xs) > 5000:
            rnd = random.Random(42)
            return [xs[rnd.randrange(len(xs))] for _ in range(5000)]
        return xs
    medA, lA_g, hA_g = bootstrap_median_ci(short(all_gapsA))
    medB, lB_g, hB_g = bootstrap_median_ci(short(all_gapsB))
    print(f"  A predictable : median={medA:6.3f}  CI [{lA_g:6.3f}, {hA_g:6.3f}]  (n={len(all_gapsA)})")
    print(f"  B creative    : median={medB:6.3f}  CI [{lB_g:6.3f}, {hB_g:6.3f}]  (n={len(all_gapsB)})")

    # tail percentile comparison
    print(f"\n=== gap percentiles (ms) ===")
    for p in (50, 75, 90, 95, 99):
        print(f"  p{p:<2}  A={q(all_gapsA, p/100):7.2f}   B={q(all_gapsB, p/100):7.2f}")

    # paired t on per-trial tok/s
    diffs = [a-b for a,b in zip(tpsA, tpsB)]
    md, ld, hd = mean_ci_t(diffs)
    print(f"\n=== paired A−B tok/s difference ===")
    print(f"  mean diff = {md:6.2f}  95% CI [{ld:6.2f}, {hd:6.2f}]  tok/s")
    if ld > 0:
        print(f"  -> CI does NOT include zero: predictable is significantly faster.")
    else:
        print(f"  -> CI includes zero: no significant difference.")

    Path("/tmp/specdec_tokens_raw.json").write_text(json.dumps(
        {"per_trial": per_trial, "model": MODEL}, indent=2))

if __name__ == "__main__":
    main()
