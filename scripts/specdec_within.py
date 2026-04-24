#!/usr/bin/env python3
"""
Within-stream speculative-decoding test.

Single prompt produces TWO sections in one response:
  SECTION A (predictable): copy a fixed JSON record 20 times
  SECTION B (creative):    a 5-sentence story about a big mouse

Because both sections come from the same request, network / queueing /
batching confounds cancel out. Any systematic difference in per-token
timing between A and B is intrinsic to the decoding stage.

We detect the A→B boundary by a marker "###STORY###".

For each section we report:
  - chunks, tokens, duration
  - median / IQR inter-chunk gap (ms)
  - chars per chunk distribution
  - burst% (chunks arriving <1ms after the prior)
"""

import json, os, sys, time, urllib.request, statistics
from pathlib import Path

MODEL = "gpt-5.4-2026-03-05"
TRIALS = 12
MAX_TOKENS = 900
KEY = os.environ["OPENAI_API_KEY"]

PROMPT = (
    'Task 1: Output the following JSON record EXACTLY 20 times, each on its own line, '
    'no numbering, no code fences, no commentary:\n'
    '{"id":1,"name":"widget","price":9.99,"in_stock":true}\n\n'
    'Task 2: After the last JSON line, output a line containing exactly "###STORY###" '
    'and nothing else. Then write a 5-sentence original story about a big mouse. '
    'Do not output anything after the story.'
)

MARKER = "###STORY###"

def stream_once(nonce: str):
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
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    t_start = time.perf_counter()
    chunks = []    # (t_rel, content)
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
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                chunks.append((time.perf_counter() - t_start, content))
    return chunks, usage

def q(xs, p):
    xs = sorted(xs); n = len(xs)
    if n == 0: return float("nan")
    k = (n-1)*p; f = int(k); c = min(f+1, n-1)
    return xs[f] + (xs[c]-xs[f])*(k-f)

def split_at_marker(chunks):
    """Split chunk list into (section_A, section_B) at MARKER.
    Discards the marker-containing chunks themselves so neither side is contaminated."""
    full = "".join(c for _, c in chunks)
    idx = full.find(MARKER)
    if idx < 0: return None, None
    # Walk chunk-by-chunk to find split index
    acc = 0
    split_before = None  # first chunk index whose content starts at/after marker
    for i, (_, c) in enumerate(chunks):
        if acc <= idx < acc + len(c):
            split_before = i; break
        acc += len(c)
    if split_before is None: return None, None
    # section A = chunks strictly before marker chunk
    # find first chunk that starts strictly after marker end
    end_idx = idx + len(MARKER)
    acc = 0; split_after = None
    for i, (_, c) in enumerate(chunks):
        if acc >= end_idx:
            split_after = i; break
        acc += len(c)
    if split_after is None: split_after = len(chunks)
    return chunks[:split_before], chunks[split_after:]

def section_stats(chunks):
    if len(chunks) < 3:
        return None
    gaps = [(chunks[i][0] - chunks[i-1][0]) * 1000.0 for i in range(1, len(chunks))]
    sizes = [len(c) for _, c in chunks]
    dur = chunks[-1][0] - chunks[0][0]
    burst = sum(1 for g in gaps if g < 1.0)
    return {
        "n_chunks": len(chunks),
        "duration_s": dur,
        "chunks_per_s": (len(chunks)-1) / max(dur, 1e-9),
        "gap_p50_ms": q(gaps, .50),
        "gap_p25_ms": q(gaps, .25),
        "gap_p75_ms": q(gaps, .75),
        "gap_p90_ms": q(gaps, .90),
        "mean_chars_per_chunk": statistics.mean(sizes),
        "burst_pct": 100.0 * burst / len(gaps),
        "gaps": gaps,
        "sizes": sizes,
    }

def main():
    print(f"model={MODEL}  trials={TRIALS}  max_tokens={MAX_TOKENS}", flush=True)
    A_stats, B_stats = [], []
    A_gaps_all, B_gaps_all = [], []

    for i in range(TRIALS):
        nonce = f"WITHIN-{i}-{time.time_ns()}"
        try:
            chunks, usage = stream_once(nonce)
        except Exception as e:
            print(f"  [{i}] ERROR: {e}", file=sys.stderr, flush=True); continue
        secA, secB = split_at_marker(chunks)
        if secA is None:
            print(f"  [{i}] marker not found; output len={sum(len(c) for _,c in chunks)}",
                  file=sys.stderr, flush=True); continue
        sA = section_stats(secA); sB = section_stats(secB)
        if not sA or not sB:
            print(f"  [{i}] section too short A={len(secA)} B={len(secB)}",
                  file=sys.stderr, flush=True); continue
        A_stats.append(sA); B_stats.append(sB)
        A_gaps_all.extend(sA["gaps"]); B_gaps_all.extend(sB["gaps"])
        print(f"  {i:02d}  A: {sA['n_chunks']:3d}ch / {sA['duration_s']:5.2f}s = "
              f"{sA['chunks_per_s']:5.1f} ch/s, gap p50={sA['gap_p50_ms']:5.1f}ms, "
              f"burst={sA['burst_pct']:4.1f}%  ||  "
              f"B: {sB['n_chunks']:3d}ch / {sB['duration_s']:5.2f}s = "
              f"{sB['chunks_per_s']:5.1f} ch/s, gap p50={sB['gap_p50_ms']:5.1f}ms, "
              f"burst={sB['burst_pct']:4.1f}%", flush=True)

    def agg(xs, key): return [x[key] for x in xs]
    print(f"\n=== SECTION A (predictable JSON) across {len(A_stats)} trials ===")
    print(f"  chunks/s         median={statistics.median(agg(A_stats,'chunks_per_s')):.2f}")
    print(f"  gap p50 ms       median={statistics.median(agg(A_stats,'gap_p50_ms')):.2f}")
    print(f"  gap p90 ms       median={statistics.median(agg(A_stats,'gap_p90_ms')):.2f}")
    print(f"  chars/chunk mean median={statistics.median(agg(A_stats,'mean_chars_per_chunk')):.2f}")
    print(f"  burst%           median={statistics.median(agg(A_stats,'burst_pct')):.2f}")

    print(f"\n=== SECTION B (creative story) across {len(B_stats)} trials ===")
    print(f"  chunks/s         median={statistics.median(agg(B_stats,'chunks_per_s')):.2f}")
    print(f"  gap p50 ms       median={statistics.median(agg(B_stats,'gap_p50_ms')):.2f}")
    print(f"  gap p90 ms       median={statistics.median(agg(B_stats,'gap_p90_ms')):.2f}")
    print(f"  chars/chunk mean median={statistics.median(agg(B_stats,'mean_chars_per_chunk')):.2f}")
    print(f"  burst%           median={statistics.median(agg(B_stats,'burst_pct')):.2f}")

    # gap histogram comparison over all gaps
    edges = [0, 0.1, 1, 5, 10, 20, 40, 80, 160, 10_000]
    def histrow(gaps, edges):
        out = [0]*(len(edges)-1)
        for g in gaps:
            for i in range(len(edges)-1):
                if edges[i] <= g < edges[i+1]:
                    out[i]+=1; break
            else:
                out[-1]+=1
        return out
    hA = histrow(A_gaps_all, edges); hB = histrow(B_gaps_all, edges)
    print(f"\n=== pooled inter-chunk gap histogram ===")
    print(f"  {'bucket (ms)':<18}  {'A n':>6}  {'A %':>6}  {'B n':>6}  {'B %':>6}")
    totA, totB = sum(hA), sum(hB)
    for i in range(len(edges)-1):
        pa = 100*hA[i]/max(totA,1); pb = 100*hB[i]/max(totB,1)
        print(f"  [{edges[i]:>6.1f},{edges[i+1]:>7.1f})  {hA[i]:>6}  {pa:5.1f}  {hB[i]:>6}  {pb:5.1f}")

    if A_stats and B_stats:
        rA = statistics.median(agg(A_stats,'chunks_per_s'))
        rB = statistics.median(agg(B_stats,'chunks_per_s'))
        print(f"\n  chunks/s ratio A/B = {rA/rB:.2f}x  (predictable faster than creative by this factor)")

    Path("/tmp/specdec_within_raw.json").write_text(json.dumps(
        {"A": A_stats, "B": B_stats}, default=list, indent=2))

if __name__ == "__main__":
    main()
