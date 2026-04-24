# Security

This repository is intentionally public-safe.

The original scratch session used local API keys, but no keys, key-test scripts, response files, or full transcripts are included here. The runnable scripts require `OPENAI_API_KEY` from the environment.

Before publishing, scan with:

```bash
rg -n 'sk-proj|OPENAI_API_KEY=|Authorization: Bearer sk-|keys\\.txt' .
```

