#!/bin/bash
# Start vertex-proxy container.
# Works with podman or docker. Reads .env for configuration.
#
# Usage:
#   ./run.sh          # start in background
#   ./run.sh stop     # stop and remove
#   ./run.sh logs     # follow logs
#   ./run.sh status   # check if running

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_NAME="vertex-proxy"
HOST_PORT="${VERTEX_PROXY_HOST_PORT:-8788}"
INTERNAL_PORT="${VERTEX_PROXY_PORT:-8780}"

# Load .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Detect runtime
if command -v podman &>/dev/null; then
    RUNTIME=podman
elif command -v docker &>/dev/null; then
    RUNTIME=docker
else
    echo "Error: neither podman nor docker found" >&2
    exit 1
fi

# ADC credentials path
ADC_DEFAULT="$HOME/.config/gcloud/application_default_credentials.json"
ADC_PATH="${GOOGLE_ADC_PATH:-$ADC_DEFAULT}"
GAC_ENV="GOOGLE_APPLICATION_CREDENTIALS"

case "${1:-start}" in
    start)
        # Stop existing if running
        $RUNTIME stop $CONTAINER_NAME 2>/dev/null || true
        $RUNTIME rm $CONTAINER_NAME 2>/dev/null || true

        # Build if image doesn't exist
        if ! $RUNTIME image exists $CONTAINER_NAME:latest 2>/dev/null; then
            echo "Building $CONTAINER_NAME image..."
            $RUNTIME build -t $CONTAINER_NAME:latest "$SCRIPT_DIR"
        fi

        echo "Starting $CONTAINER_NAME on port $HOST_PORT..."
        $RUNTIME run -d --name $CONTAINER_NAME \
            -p "$HOST_PORT:$INTERNAL_PORT" \
            -e VERTEX_PROXY_PROJECT_ID="${VERTEX_PROXY_PROJECT_ID:-}" \
            -e VERTEX_PROXY_ANTHROPIC_REGION="${VERTEX_PROXY_ANTHROPIC_REGION:-global}" \
            -e VERTEX_PROXY_GEMINI_REGION="${VERTEX_PROXY_GEMINI_REGION:-us-central1}" \
            -e VERTEX_PROXY_EMBEDDING_REGION="${VERTEX_PROXY_EMBEDDING_REGION:-global}" \
            -e VERTEX_PROXY_MAAS_REGION="${VERTEX_PROXY_MAAS_REGION:-us-central1}" \
            -e VERTEX_PROXY_HOST=0.0.0.0 \
            -e VERTEX_PROXY_PORT="$INTERNAL_PORT" \
            -e VERTEX_PROXY_LOG_LEVEL="${VERTEX_PROXY_LOG_LEVEL:-info}" \
            -e VERTEX_PROXY_METRICS_ENABLED="${VERTEX_PROXY_METRICS_ENABLED:-false}" \
            -e $GAC_ENV=/run/secrets/gcloud/adc.json \
            -v "$ADC_PATH:/run/secrets/gcloud/adc.json:ro" \
            --restart unless-stopped \
            $CONTAINER_NAME:latest

        echo "Waiting for health check..."
        for i in $(seq 1 15); do
            if curl -sf "http://localhost:$HOST_PORT/health" >/dev/null 2>&1; then
                echo "vertex-proxy is healthy on port $HOST_PORT"
                exit 0
            fi
            sleep 1
        done
        echo "Warning: health check timed out. Check logs with: $0 logs"
        ;;
    stop)
        $RUNTIME stop $CONTAINER_NAME 2>/dev/null && echo "Stopped" || echo "Not running"
        $RUNTIME rm $CONTAINER_NAME 2>/dev/null || true
        ;;
    logs)
        $RUNTIME logs -f $CONTAINER_NAME
        ;;
    status)
        $RUNTIME inspect $CONTAINER_NAME --format '{{.State.Status}}' 2>/dev/null || echo "not running"
        ;;
    rebuild)
        $RUNTIME stop $CONTAINER_NAME 2>/dev/null || true
        $RUNTIME rm $CONTAINER_NAME 2>/dev/null || true
        $RUNTIME rmi $CONTAINER_NAME:latest 2>/dev/null || true
        $RUNTIME build -t $CONTAINER_NAME:latest "$SCRIPT_DIR"
        exec "$0" start
        ;;
    *)
        echo "Usage: $0 {start|stop|logs|status|rebuild}"
        exit 1
        ;;
esac
