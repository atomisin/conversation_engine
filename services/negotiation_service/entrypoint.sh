#!/usr/bin/env sh
set -e

# POSIX-safe entrypoint for negotiation service
# - loads secrets from /run/secrets
# - prints masked diagnostics (using eval to read variable by name)
# - tests language detection and remote LLM (best-effort)
# - ALWAYS exec "$@" at the end so CMD runs (uvicorn)

_read_secret_file() {
  file_path="$1"
  env_name="$2"

  if [ -f "$file_path" ]; then
    val="$(tr -d '\r\n' < "$file_path")"
    export "$env_name=$val"
    echo "[entrypoint] Loaded ${env_name} from ${file_path}"
  else
    echo "[entrypoint] Warning: Secret file ${file_path} not found"
  fi
}

# Load secrets
_read_secret_file "/run/secrets/groq_api_key" "GROQ_API_KEY"
_read_secret_file "/run/secrets/openai_api_key" "OPENAI_API_KEY"
_read_secret_file "/run/secrets/hf_token" "HF_TOKEN"
_read_secret_file "/run/secrets/llm_api_key" "LLM_API_KEY"

# Fallback if provider-specific keys are missing
if [ -z "${GROQ_API_KEY:-}" ] && [ -n "${LLM_API_KEY:-}" ]; then
  export GROQ_API_KEY="${LLM_API_KEY}"
fi
if [ -z "${OPENAI_API_KEY:-}" ] && [ -n "${LLM_API_KEY:-}" ]; then
  export OPENAI_API_KEY="${LLM_API_KEY}"
fi

# Masked env print helper (POSIX-safe; uses eval to read variable by name)
print_masked() {
  keyval="$1"
  # get the value of the variable whose name is in $keyval
  val=$(eval "printf '%s' \"\${$keyval}\"" 2>/dev/null || true)
  if [ -n "${val:-}" ]; then
    HEAD=$(printf "%s" "$val" | cut -c1-6)
    TAIL=$(printf "%s" "$val" | rev | cut -c1-4 | rev)
    printf "%s=%s...%s\n" "$keyval" "$HEAD" "$TAIL"
  else
    printf "%s=<missing>\n" "$keyval"
  fi
}

echo "[entrypoint] Final environment (masked):"
print_masked GROQ_API_KEY
print_masked OPENAI_API_KEY
print_masked HF_TOKEN
print_masked LLM_API_KEY
printf "LLM_REMOTE_URL=%s\n" "${LLM_REMOTE_URL:-<unset>}"
printf "LLM_REMOTE_PROVIDER=%s\n" "${LLM_REMOTE_PROVIDER:-<unset>}"
printf "LLM_MODEL=%s\n" "${LLM_MODEL:-<unset>}"
printf "LLM_REMOTE_TIMEOUT=%s\n" "${LLM_REMOTE_TIMEOUT:-10}"

: "${LLM_REMOTE_TIMEOUT:=10}"

# Language detection test (best-effort)
if [ -n "${LANGUAGE_DETECTION_URL:-}" ]; then
  echo "[entrypoint] Testing language detection service..."
  if curl -sS -X POST "${LANGUAGE_DETECTION_URL}" \
      -H "Content-Type: application/json" \
      -d '{"text":"Hello"}' --max-time 3 >/dev/null 2>&1; then
    echo "[entrypoint] Language detection reachable"
  else
    echo "[entrypoint] Language detection test failed"
  fi
fi

# Remote LLM test (best-effort)
if [ -n "${LLM_REMOTE_URL:-}" ] || [ -n "${GROQ_API_KEY:-}" ] || [ -n "${OPENAI_API_KEY:-}" ]; then
  echo "[entrypoint] Testing remote LLM endpoint..."
  PROVIDER="${LLM_REMOTE_PROVIDER:-GROQ}"
  URL="${LLM_REMOTE_URL:-https://api.groq.com/openai/v1/chat/completions}"

  if [ -n "${GROQ_API_KEY:-}" ]; then
    KEY="${GROQ_API_KEY}"
  elif [ -n "${OPENAI_API_KEY:-}" ]; then
    KEY="${OPENAI_API_KEY}"
  elif [ -n "${LLM_API_KEY:-}" ]; then
    KEY="${LLM_API_KEY}"
  else
    KEY=""
  fi

  if [ -z "${KEY}" ]; then
    echo "[entrypoint] No API key available for remote LLM test; skipping."
  else
    MASK_HEAD=$(printf "%s" "$KEY" | cut -c1-6)
    MASK_TAIL=$(printf "%s" "$KEY" | rev | cut -c1-4 | rev)
    MODEL="${LLM_MODEL:-groq/compound-mini}"
    echo "[entrypoint] Testing URL=${URL} provider=${PROVIDER} model=${MODEL} (key ${MASK_HEAD}...${MASK_TAIL})"

    # Groq/OpenAI-compatible chat payload (system + user)
    body='{"model":"'"${MODEL}"'","messages":[{"role":"system","content":"You are a polite Nigerian market seller."},{"role":"user","content":"say hi"}],"max_tokens":5}'
    http_output=$(curl -sS -X POST "${URL}" \
      -H "Authorization: Bearer ${KEY}" \
      -H "Content-Type: application/json" \
      -d "${body}" --max-time "${LLM_REMOTE_TIMEOUT}" -w "\n%{http_code}" 2>/dev/null || true)

    http_code=$(printf "%s" "$http_output" | tail -n1 || echo "")
    body_preview=$(printf "%s" "$http_output" | sed '$d' | head -c 1000 || echo "")
    if [ -n "$body_preview" ]; then
      SHORT_PREVIEW=$(printf "%s" "$body_preview" | tr '\n' ' ' | cut -c1-400)
    else
      SHORT_PREVIEW=""
    fi
    echo "[entrypoint] Remote LLM test returned HTTP ${http_code} preview: ${SHORT_PREVIEW}"
  fi
fi

echo "[entrypoint] Starting app..."
exec "$@"
