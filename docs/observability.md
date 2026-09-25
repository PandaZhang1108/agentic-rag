# Observability and replay

The service can send LangGraph traces to a self-hosted Langfuse instance. Each request records model and tool spans, latency, token usage when the provider returns it, and estimated cost based on the configured price table.

Langfuse is optional and fail-open: if tracing is disabled or unavailable, the chat request continues. Provider billing remains the source of truth because cache behavior and prices can change.

For failure analysis, `scripts/replay_generation.py` reuses a fixed intermediate input and reruns only the generation stage. This separates generation problems from retrieval changes and makes prompt or model comparisons easier to audit.

Start the local Langfuse stack:

```bash
bash scripts/langfuse_local.sh up
```

Then set `LANGFUSE_ENABLED=true` and the generated public and secret keys in `.env`.
