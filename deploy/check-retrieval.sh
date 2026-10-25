#!/usr/bin/env bash
# Sanity check for the retrieval pair (and a template for the author's acceptance test:
# run it against a reference engine and diff the numbers).
#   EMBED_URL / RERANK_URL override the endpoints.
set -euo pipefail
EMBED_URL="${EMBED_URL:-http://127.0.0.1:8004}"
RERANK_URL="${RERANK_URL:-http://127.0.0.1:8005}"

echo "== embedding =="
curl -sf "$EMBED_URL/v1/embeddings" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding","input":"Beijing is the capital of China."}' \
| python3 -c 'import json,sys; d=json.load(sys.stdin); v=d["data"][0]["embedding"]; print("dims:", len(v), "| first3:", [round(x,4) for x in v[:3]])'

echo "== rerank =="
curl -sf "$RERANK_URL/v1/rerank" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-reranker","query":"What is the capital of China?","documents":["Beijing is the capital of China.","The Eiffel Tower is in Paris."],"top_n":2}' \
| python3 -c 'import json,sys; d=json.load(sys.stdin); [print("doc", r["index"], "score", round(r["relevance_score"],4)) for r in d["results"]]'
