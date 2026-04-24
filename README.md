# Speculative Decoding Fingerprint Experiment

This repo contains a public-safe reconstruction of a black-box experiment for detecting speculative-decoding-family acceleration in the OpenAI GPT API streaming path.

The tested model was `gpt-5.4-2026-03-05`. The experiment does not prove the exact internal algorithm. It shows a strong content-dependent decoding-speed fingerprint consistent with speculative decoding, lookahead decoding, Medusa-style decoding, prompt-lookup decoding, or another batched acceptance scheme.

## Result

### Main within-stream token test

One streamed request generated two sections in sequence:

1. Predictable: repeat a fixed JSON record exactly 20 times.
2. Creative: write a 5-sentence story.

The sections share one HTTP stream, one TCP connection, one model invocation, and one server-side context. The measurement excludes time-to-first-token and compares only generation-phase token throughput.

| Section | Mean tok/s | 95% CI |
| --- | ---: | ---: |
| Predictable JSON | 69.76 | [67.96, 71.56] |
| Creative story | 36.42 | [34.72, 38.13] |
| Ratio JSON/story | 1.933x | [1.832, 2.034] |

Predictable output was faster in 20/20 paired trials. The paired difference was +33.33 tok/s with 95% CI [30.85, 35.82].

### Order-control test

To rule out "first section is faster" or warm-up effects, the order was randomized across two prompt variants:

| Order | JSON tok/s | Story tok/s | Ratio |
| --- | ---: | ---: | ---: |
| FWD, JSON then story | 76.10 [72.00, 80.20] | 40.65 [37.69, 43.62] | 1.898 [1.742, 2.053] |
| REV, story then JSON | 86.92 [82.85, 91.00] | 30.42 [29.19, 31.65] | 2.859 [2.766, 2.952] |

JSON was faster in 30/30 order-control trials, regardless of which section came first.

## Interpretation

A plain autoregressive decoder should be largely content-agnostic at the token generation stage: one target-model forward pass per emitted token, independent of whether the text is easy JSON repetition or creative prose.

The observed 1.9-2.9x speed difference, with tight confidence intervals and order invariance, strongly suggests content-dependent acceleration in the serving path. That is the expected fingerprint of speculative-decoding-family methods: predictable spans have high acceptance rates and emit more tokens per expensive verification step; creative spans reject more often and stall.

## Limitations

- This is a black-box timing experiment. It cannot distinguish draft-model speculative decoding from lookahead, Medusa, prompt lookup, or another verification/acceptance variant.
- It was run on one model snapshot, one account, one time window, and likely one region.
- Tokenization used `tiktoken` `o200k_base`. The vocabulary check found a constant -3 token offset versus API `completion_tokens` on the ASCII-heavy test strings, which does not affect throughput ratios.
- Network jitter is reduced by the within-stream design, but not eliminated.

## Files

- `scripts/specdec_tokens.py` - main within-stream token-level experiment.
- `scripts/specdec_ordered.py` - order-control experiment.
- `scripts/specdec_within.py` - earlier within-stream chunk-level experiment.
- `scripts/specdec_chunk_gaps.py` - earlier inter-chunk gap/burst analysis.
- `scripts/vocab_check.py` - tokenizer sanity check against API usage counts.
- `results/specdec_tokens_log.txt` - successful main token-level run.
- `results/specdec_ordered_log.txt` - successful order-control run.
- `results/vocab_check_log.txt` - tokenizer sanity-check output.
- `results/summary.json` - compact machine-readable result summary.

## Running

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY='...'
python scripts/specdec_tokens.py
python scripts/specdec_ordered.py
```

The scripts intentionally read the API key from `OPENAI_API_KEY`. No keys are included in this repo.

