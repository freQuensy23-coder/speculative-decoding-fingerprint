#!/usr/bin/env python3
"""
Ordering-controlled speculative-decoding test.

Two prompt orderings, each run N trials, interleaved:
  FWD : JSON-repeat first,  then story
  REV : story first,        then JSON-repeat

If the predictable/creative ratio is driven by CONTENT, we expect
tok/s(JSON) > tok/s(story) in BOTH orderings.
If it's driven by POSITION (e.g. model warms up), we expect the
first section to be faster regardless of content.

Marker "###SPLIT###" separates sections.
"""
import json, os, sys, time, math, statistics, random, urllib.request
from pathlib import Path
import tiktoken

MODEL = "gpt-5.4-2026-03-05"
TRIALS_PER_ORDER = 15
MAX_TOKENS = 900
KEY = os.environ["OPENAI_API_KEY"]
MARKER = "###SPLIT###"
ENC = tiktoken.get_encoding("o200k_base")

JSON_TASK = (
    'Output the following JSON record EXACTLY 20 times, each on its own line, '
    'no numbering, no code fences, no commentary:\n'
    '{"id":1,"name":"widget","price":9.99,"in_stock":true}'
)
STORY_TASK = 'Write a 5-sentence original story about a big mouse.'

def build_prompt(order: str) -> str:
    sep = f'\nAfter completing the previous task, output a line containing exactly "{MARKER}" and nothing else, then proceed.\n'
    if order == "FWD":
        return f'Task 1: {JSON_TASK}\n{sep}\nTask 2: {STORY_TASK}\nDo not output anything after the story.'
    else:  # REV
        return f'Task 1: {STORY_TASK}\n{sep}\nTask 2: {JSON_TASK}\nDo not output anything after the JSON block.'

def stream_once(prompt, nonce):
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
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json",
                 "Accept": "text/event-stream"},
        method="POST",
    )
    t_start = time.perf_counter()
    char_times, text_parts = [], []
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
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                text_parts.append(delta)
                char_times.extend([now]*len(delta))
    full = "".join(text_parts)
    ids = ENC.encode(full, disallowed_special=())
    tokens = []
    pos = 0
    for tid in ids:
        piece = ENC.decode([tid])
        end = pos + len(piece)
        t = char_times[end-1] if end-1 < len(char_times) else (char_times[-1] if char_times else 0.0)
        tokens.append({"t": t, "tok": piece})
        pos = end
    return tokens, usage, full

def split_at_marker(tokens, full):
    idx = full.find(MARKER)
    if idx < 0: return None, None
    acc = 0; first = last = None
    for i, tk in enumerate(tokens):
        start = acc; end = acc + len(tk["tok"])
        if end > idx and start < idx + len(MARKER):
            if first is None: first = i
            last = i
        acc = end
    if first is None: return None, None
    return tokens[:first], tokens[last+1:]

def tps(sec):
    if len(sec) < 3: return None
    dur = sec[-1]["t"] - sec[0]["t"]
    return (len(sec)-1) / max(dur, 1e-9)

def mean_ci_t(xs):
    n = len(xs)
    if n < 2: return (statistics.mean(xs) if xs else float("nan"), 0.0, 0.0)
    m = statistics.mean(xs); sd = statistics.stdev(xs); se = sd/math.sqrt(n)
    tt = {2:12.71,3:4.30,4:3.18,5:2.78,6:2.57,7:2.45,8:2.36,9:2.31,10:2.26,
          11:2.23,12:2.20,13:2.18,14:2.16,15:2.14,16:2.13,17:2.12,18:2.11,
          19:2.10,20:2.09,25:2.06,30:2.04}
    tc = tt.get(n-1, 2.0)
    return m, m-tc*se, m+tc*se

def q(xs, p):
    xs = sorted(xs); n = len(xs)
    if n == 0: return float("nan")
    k = (n-1)*p; f = int(k); c = min(f+1, n-1)
    return xs[f] + (xs[c]-xs[f])*(k-f)

def main():
    print(f"model={MODEL}  trials/order={TRIALS_PER_ORDER}  max_tokens={MAX_TOKENS}", flush=True)
    # rows: order, trial_idx, json_tps, story_tps, json_is_first, json_pos_in_stream
    rows = []
    rnd = random.Random(0xBEEF)
    schedule = []
    for i in range(TRIALS_PER_ORDER):
        schedule.append(("FWD", i))
        schedule.append(("REV", i))
    rnd.shuffle(schedule)  # interleave randomly to spread any load drift

    for order, i in schedule:
        nonce = f"{order}-{i}-{time.time_ns()}"
        prompt = build_prompt(order)
        try:
            tokens, usage, full = stream_once(prompt, nonce)
        except Exception as e:
            print(f"  [{order} {i}] ERROR: {e}", file=sys.stderr, flush=True); continue
        sec1, sec2 = split_at_marker(tokens, full)
        if sec1 is None:
            print(f"  [{order} {i}] no marker (len={len(full)})", file=sys.stderr, flush=True); continue
        if order == "FWD":
            json_sec, story_sec = sec1, sec2
        else:
            story_sec, json_sec = sec1, sec2
        tJ = tps(json_sec); tS = tps(story_sec)
        if tJ is None or tS is None:
            print(f"  [{order} {i}] short section J={len(json_sec)} S={len(story_sec)}",
                  file=sys.stderr, flush=True); continue
        rows.append({"order": order, "trial": i,
                     "json_tps": tJ, "story_tps": tS,
                     "n_json": len(json_sec), "n_story": len(story_sec)})
        print(f"  {order} {i:02d}  JSON: {len(json_sec):3d}tok {tJ:5.1f} tok/s  |  "
              f"STORY: {len(story_sec):3d}tok {tS:5.1f} tok/s  |  ratio J/S={tJ/tS:.2f}",
              flush=True)

    if not rows:
        print("no valid rows"); return

    # summarize by order
    print()
    for order in ("FWD", "REV"):
        sub = [r for r in rows if r["order"] == order]
        if not sub: continue
        J = [r["json_tps"] for r in sub]; S = [r["story_tps"] for r in sub]
        R = [j/s for j,s in zip(J,S)]
        mJ, lJ, hJ = mean_ci_t(J); mS, lS, hS = mean_ci_t(S); mR, lR, hR = mean_ci_t(R)
        n_above = sum(1 for r in R if r > 1.0)
        first = "JSON" if order == "FWD" else "STORY"
        print(f"=== ORDER={order}  (first section = {first})  n={len(sub)} ===")
        print(f"  JSON   tok/s  mean={mJ:6.2f}  95% CI [{lJ:6.2f}, {hJ:6.2f}]")
        print(f"  STORY  tok/s  mean={mS:6.2f}  95% CI [{lS:6.2f}, {hS:6.2f}]")
        print(f"  ratio  JSON/STORY  mean={mR:5.3f}  95% CI [{lR:5.3f}, {hR:5.3f}]")
        print(f"  trials with JSON faster than STORY: {n_above}/{len(R)}")
        print()

    # content vs position test:
    # If CONTENT drives speed: JSON > STORY in both orders (ratios >>1 in both)
    # If POSITION drives speed: first section > second. So in FWD (JSON first), ratio J/S>1;
    #   but in REV (STORY first), ratio J/S<1.
    fwd_ratios = [r["json_tps"]/r["story_tps"] for r in rows if r["order"]=="FWD"]
    rev_ratios = [r["json_tps"]/r["story_tps"] for r in rows if r["order"]=="REV"]
    if fwd_ratios and rev_ratios:
        mF, lF, hF = mean_ci_t(fwd_ratios); mR, lR, hR = mean_ci_t(rev_ratios)
        print("=== content vs position verdict ===")
        print(f"  FWD (JSON first)  ratio J/S = {mF:.3f} [{lF:.3f}, {hF:.3f}]")
        print(f"  REV (STORY first) ratio J/S = {mR:.3f} [{lR:.3f}, {hR:.3f}]")
        if lF > 1.0 and lR > 1.0:
            print("  -> J/S > 1 in BOTH orders. Result is CONTENT-driven, not position.")
        elif hF < 1.0 or hR < 1.0 or (lF < 1.0 and lR < 1.0):
            print("  -> direction flips with order. Result is POSITION-driven.")
        else:
            print("  -> ambiguous.")

    Path("/tmp/specdec_ordered_raw.json").write_text(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
