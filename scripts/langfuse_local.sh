#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/observability/langfuse/docker-compose.yml"
ENV_FILE="$ROOT_DIR/.env.langfuse"
PROJECT_NAME="agentic-rag-langfuse"

prepare_public_images() {
  # Docker Desktop 的 macOS Keychain 偶尔会阻止公开镜像拉取。缺少镜像时
  # 用一个没有 credential helper 的临时配置拉取；已有镜像不会重复下载。
  local public_config="${TMPDIR:-/tmp}/agentic-rag-docker-public"
  mkdir -p "$public_config"
  printf '{}\n' > "$public_config/config.json"
  local images=(
    "docker.io/redis:7"
    "docker.io/postgres:17"
    "docker.io/clickhouse/clickhouse-server:25.12"
    "cgr.dev/chainguard/minio:latest"
    "docker.langfuse.com/langfuse/langfuse:4"
    "docker.langfuse.com/langfuse/langfuse-worker:4"
  )
  local image
  for image in "${images[@]}"; do
    if ! docker image inspect "$image" >/dev/null 2>&1; then
      docker --config "$public_config" pull "$image"
    fi
  done
}

if [[ ! -f "$ENV_FILE" ]]; then
  echo "缺少 $ENV_FILE；请先完成 Langfuse 本地配置。" >&2
  exit 1
fi

case "${1:-}" in
  up)
    prepare_public_images
    docker compose --project-name "$PROJECT_NAME" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d
    for _ in {1..30}; do
      if curl -fsS http://127.0.0.1:3000/api/public/health >/dev/null 2>&1; then
        if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
          "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/configure_langfuse_pricing.py"
        fi
        break
      fi
      sleep 2
    done
    ;;
  down)
    docker compose --project-name "$PROJECT_NAME" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down
    ;;
  status)
    docker compose --project-name "$PROJECT_NAME" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps
    ;;
  logs)
    docker compose --project-name "$PROJECT_NAME" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" logs --tail=200 "${2:-langfuse-web}"
    ;;
  *)
    echo "用法：$0 {up|down|status|logs [service]}" >&2
    exit 2
    ;;
esac
