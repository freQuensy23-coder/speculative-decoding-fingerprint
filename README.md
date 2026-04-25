# GPT speculative decoding fingerprint

Black-box timing tests for content-dependent decoding speed in the OpenAI streaming API.

Model tested: `gpt-5.4-2026-03-05`.

This does not identify OpenAI's serving algorithm. It shows that predictable output was generated much faster than less predictable output inside the same stream, which is the timing pattern expected from speculative-decoding-family serving methods.

## Results

### Within-stream test

One request produced two sections:

- fixed JSON record repeated 20 times
- 5-sentence story

The marker between sections was excluded. Timing starts at the first token of each section, so TTFT, queueing, prefill, and network setup are not part of the measurement.

| Section | Mean tok/s | 95% CI |
| --- | ---: | ---: |
| JSON | 69.76 | [67.96, 71.56] |
| Story | 36.42 | [34.72, 38.13] |
| JSON/story | 1.933x | [1.832, 2.034] |

JSON was faster in 20/20 paired trials. Paired difference: +33.33 tok/s, 95% CI [30.85, 35.82].

Full log: `results/specdec_tokens_log.txt`.

### Order control

The second test randomizes two prompt orders:

- FWD: JSON, then story
- REV: story, then JSON

| Order | JSON tok/s | Story tok/s | Ratio |
| --- | ---: | ---: | ---: |
| FWD | 76.10 [72.00, 80.20] | 40.65 [37.69, 43.62] | 1.898 [1.742, 2.053] |
| REV | 86.92 [82.85, 91.00] | 30.42 [29.19, 31.65] | 2.859 [2.766, 2.952] |

JSON was faster in 30/30 trials, so the effect is content-driven rather than "first section is faster".

Full log: `results/specdec_ordered_log.txt`.

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...

python scripts/specdec_tokens.py
python scripts/specdec_ordered.py
```

Both scripts write raw JSON to `/tmp` by default. Change `--out` to keep raw output elsewhere.

## Files

- `scripts/specdec_tokens.py` - main within-stream test
- `scripts/specdec_ordered.py` - order control
- `results/summary.json` - compact result summary
- `results/specdec_tokens_log.txt` - main run output
- `results/specdec_ordered_log.txt` - order-control run output

## Caveats

- Black-box timing cannot distinguish draft-model speculative decoding from lookahead, prompt lookup, Medusa-style decoding, or another acceptance scheme.
- The run used one model snapshot, one account, and one time window.
- Token timestamps are reconstructed from streamed text chunks and `tiktoken` `o200k_base`.
