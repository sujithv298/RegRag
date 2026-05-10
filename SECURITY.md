# Security

## API key handling

This template integrates with cloud LLM providers (Anthropic, OpenAI) that require API keys. The keys you use are your responsibility; this section documents the protocol the template assumes.

### How keys are read

The default adapters read keys from environment variables:

| Adapter            | Env var               |
|--------------------|-----------------------|
| `AnthropicAdapter` | `ANTHROPIC_API_KEY`   |
| `OpenAIAdapter`    | `OPENAI_API_KEY`      |
| `LlamaLocalAdapter`| (no key — on-prem)    |

You can also pass `api_key=` explicitly to the adapter constructor, but the env-var path is preferred so the key never appears in source code.

### Where to put your key

1. **Local development:** copy `.env.example` to `.env` and fill in the values. `.env` is in `.gitignore` and `python-dotenv` loads it automatically when the CLI starts.
2. **CI / production:** inject env vars through your orchestrator's secrets store. Never bake keys into images or config files in version control.

### What never gets logged

The audit log (`audit.jsonl`) records `model_name`, `model_version`, response text, retrieved chunk IDs, and SHA-256 hashes of the prompt and response. **It does not record the API key.** Keys live only in HTTP `Authorization` headers and are not part of any prompt or response payload.

### Pre-commit secrets scanner

`detect-secrets` runs as a pre-commit hook (`.pre-commit-config.yaml`). To set it up after cloning:

```bash
uv sync --extra dev
pre-commit install
detect-secrets scan > .secrets.baseline
```

The baseline is committed; the scanner refuses to let new high-entropy strings or known key patterns past `git commit`.

### If a key gets committed

A key that ever appeared in a commit — **even a deleted commit** — is permanently compromised. Git history is recoverable; force-pushing the deletion does not remove the key from anyone who already cloned. The protocol is:

1. **Rotate immediately.** Revoke the old key in the provider's console; issue a new one.
2. **Do not** try to scrub history (rewrite, force-push, BFG). Rotation is the fix; scrubbing is not.
3. Audit usage logs in the provider's console for the period the key was exposed.
4. Update `.secrets.baseline` once the new key is in place.

### Things to be paranoid about

- **Screenshots and demo videos.** A terminal showing `export ANTHROPIC_API_KEY=...` is a leak. Never demo with a real key visible.
- **Slack / chat.** Pasting a key into any conversation logs it in that system's storage. Don't.
- **CI logs.** If a CI step echoes env vars (e.g., `env` for debugging), it can leak the key into pipeline logs.
- **`.env.local` and similar.** Common alternative names — check they're in `.gitignore`. (They are in this template.)

## Data residency

If your environment forbids sending queries to a cloud LLM, swap the cloud adapter for `LlamaLocalAdapter` (`uv sync --extra local`). The pipeline runs unchanged on-prem.
