#!/usr/bin/env bash

set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-compose.academic.yml}"
API_URL="${ACADEMIC_API_URL:-http://127.0.0.1:8001}"
OLLAMA_MODEL="${ACADEMIC_MODEL:-qwen3.5:9b}"
E2E_PDF_PATH="${ACADEMIC_E2E_PDF_PATH:-/Users/mia/Desktop/2410.02902v4.pdf}"


compose() {
    docker compose -f "${COMPOSE_FILE}" "$@"
}


wait_for_api() {
    echo "[academic] waiting for API..."

    for attempt in $(seq 1 30); do
        if curl -fsS "${API_URL}/health/ready" >/dev/null 2>&1; then
            echo "[academic] API is healthy"
            return 0
        fi

        echo "[academic] API not ready (${attempt}/30)"
        sleep 2
    done

    echo "[academic] ERROR: API did not become healthy"
    compose ps
    compose logs --tail=100 academic-api
    return 1
}


check_ollama_from_worker() {
    echo "[academic] checking Worker -> Ollama connection..."

    compose exec -T academic-worker \
        python - "${OLLAMA_MODEL}" <<'PY'
import json
import sys
import urllib.request

required_model = sys.argv[1]
url = "http://host.docker.internal:11434/api/tags"

with urllib.request.urlopen(url, timeout=10) as response:
    payload = json.load(response)

models = [
    item.get("name", "")
    for item in payload.get("models", [])
]

print("[academic] Ollama connection OK")
print("[academic] available models:")

for model in models:
    print(f"  - {model}")

if required_model not in models:
    raise SystemExit(
        f"[academic] ERROR: required model not found: "
        f"{required_model}"
    )

print(
    f"[academic] required model available: "
    f"{required_model}"
)
PY
}


run_smoke() {
    echo "============================================================"
    echo "Academic Stack Smoke Test"
    echo "============================================================"

    compose ps

    echo
    echo "[academic] checking API health..."

    curl -fsS "${API_URL}/health/ready"
    echo

    check_ollama_from_worker

    echo
    echo "============================================================"
    echo "[academic] SMOKE TEST PASSED"
    echo "============================================================"
}


run_e2e() {
    if [[ ! -f "${E2E_PDF_PATH}" ]]; then
        echo "[academic] ERROR: PDF not found:"
        echo "${E2E_PDF_PATH}"
        exit 1
    fi

    wait_for_api

    echo "[academic] running real E2E..."

    RUN_ACADEMIC_E2E=1 \
    ACADEMIC_E2E_PDF_PATH="${E2E_PDF_PATH}" \
        python -m pytest \
        tests/academic/test_academic_api_e2e.py \
        -v -s
}


command="${1:-help}"

case "${command}" in

    build)
        echo "[academic] building Academic image..."
        compose build academic-api
        ;;

    up)
        echo "[academic] starting stack..."
        compose up -d
        wait_for_api
        compose ps
        ;;

    down)
        echo "[academic] stopping stack..."
        compose down
        ;;

    restart)
        echo "[academic] restarting stack..."
        compose down
        compose up -d
        wait_for_api
        compose ps
        ;;

    status)
        compose ps
        ;;

    logs)
        compose logs \
            --tail=100 \
            -f \
            academic-migrate \
            academic-api \
            academic-worker
        ;;

    smoke)
        wait_for_api
        run_smoke
        ;;

    e2e)
        run_e2e
        ;;

    full-check)
        wait_for_api
        run_smoke
        run_e2e
        ;;

    observability-up)
        echo "[academic] starting Prometheus and Grafana..."
        compose --profile observability up -d prometheus grafana
        compose --profile observability ps
        ;;

    observability-down)
        echo "[academic] stopping Prometheus and Grafana..."
        compose --profile observability stop prometheus grafana
        ;;

    *)
        cat <<'HELP'
Usage:

  ./scripts/academic_stack.sh build
  ./scripts/academic_stack.sh up
  ./scripts/academic_stack.sh down
  ./scripts/academic_stack.sh restart
  ./scripts/academic_stack.sh status
  ./scripts/academic_stack.sh logs
  ./scripts/academic_stack.sh smoke
  ./scripts/academic_stack.sh e2e
  ./scripts/academic_stack.sh full-check
  ./scripts/academic_stack.sh observability-up
  ./scripts/academic_stack.sh observability-down

Main workflow:

  up
    Start MySQL + Redis + API + Worker.

  smoke
    Check Compose services, API health and Worker -> Ollama.

  e2e
    Run the real Academic API end-to-end test.

  full-check
    Run smoke + real E2E together.

  observability-up
    Start the optional Prometheus + Grafana profile.

  observability-down
    Stop the optional observability services without deleting their data.
HELP
        ;;

esac
