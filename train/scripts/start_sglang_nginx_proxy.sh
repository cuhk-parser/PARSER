#!/usr/bin/env bash
set -euo pipefail

# Local nginx proxy for LongMas subagents:
# trainer -> Unix domain socket -> nginx -> TCP keepalive -> sglang.
#
# Environment knobs:
#   LONGMAS_NGINX_SOCKET=/tmp/longmas_sglang_nginx.sock
#   LONGMAS_SGLANG_UPSTREAMS=host1:port1[,host2:port2]
#   LONGMAS_NGINX_KEEPALIVE=2048
#   LONGMAS_NGINX_WORKER_CONNECTIONS=65535
#   LONGMAS_NGINX_WORKER_RLIMIT_NOFILE=65535
#   LONGMAS_NGINX_CONF=/tmp/longmas_sglang_nginx.conf
#   LONGMAS_NGINX_PID=/tmp/longmas_sglang_nginx.pid
#   LONGMAS_NGINX_BIN=/path/to/nginx

SOCKET_PATH="${LONGMAS_NGINX_SOCKET:-/tmp/longmas_sglang_nginx.sock}"
UPSTREAMS="${LONGMAS_SGLANG_UPSTREAMS:-127.0.0.1:8012}"
KEEPALIVE="${LONGMAS_NGINX_KEEPALIVE:-2048}"
WORKER_CONNECTIONS="${LONGMAS_NGINX_WORKER_CONNECTIONS:-65535}"
WORKER_RLIMIT_NOFILE="${LONGMAS_NGINX_WORKER_RLIMIT_NOFILE:-65535}"
CONF_PATH="${LONGMAS_NGINX_CONF:-/tmp/longmas_sglang_nginx.conf}"
PID_PATH="${LONGMAS_NGINX_PID:-/tmp/longmas_sglang_nginx.pid}"
ERROR_LOG="${LONGMAS_NGINX_ERROR_LOG:-/tmp/longmas_sglang_nginx_error.log}"
ACCESS_LOG="${LONGMAS_NGINX_ACCESS_LOG:-off}"
NGINX_BIN="${LONGMAS_NGINX_BIN:-$(command -v nginx || true)}"

require_nginx() {
    if [[ -z "$NGINX_BIN" || ! -x "$NGINX_BIN" ]]; then
        cat >&2 <<'EOF'
nginx executable not found.

Install nginx on the trainer node, load the cluster nginx module, or set:
  LONGMAS_NGINX_BIN=/absolute/path/to/nginx

For example, if nginx is installed in a conda environment:
  LONGMAS_NGINX_BIN=$CONDA_PREFIX/sbin/nginx
EOF
        exit 127
    fi
}

write_config() {
    local upstream_servers=""
    IFS=',' read -ra servers <<< "$UPSTREAMS"
    for server in "${servers[@]}"; do
        server="$(echo "$server" | xargs)"
        if [[ -n "$server" ]]; then
            upstream_servers+="        server ${server} max_fails=3 fail_timeout=10s;"$'\n'
        fi
    done

    if [[ -z "$upstream_servers" ]]; then
        echo "No sglang upstream configured in LONGMAS_SGLANG_UPSTREAMS" >&2
        exit 1
    fi

    cat > "$CONF_PATH" <<EOF
worker_processes auto;
worker_rlimit_nofile ${WORKER_RLIMIT_NOFILE};
error_log ${ERROR_LOG} warn;
pid ${PID_PATH};

events {
    worker_connections ${WORKER_CONNECTIONS};
}

http {
    access_log ${ACCESS_LOG};

    upstream sglang_backend {
        least_conn;
${upstream_servers}        keepalive ${KEEPALIVE};
    }

    server {
        listen unix:${SOCKET_PATH};

        location /v1/ {
            proxy_pass http://sglang_backend/v1/;

            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host \$host;

            proxy_connect_timeout 5s;
            proxy_send_timeout 600s;
            proxy_read_timeout 600s;
            proxy_buffering off;

            proxy_next_upstream error timeout http_502 http_503 http_504;
            proxy_next_upstream_tries 2;
        }
    }
}
EOF
}

start() {
    require_nginx

    if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
        echo "nginx proxy is already running with pid $(cat "$PID_PATH")"
        return 0
    fi

    mkdir -p "$(dirname "$SOCKET_PATH")" "$(dirname "$CONF_PATH")" "$(dirname "$PID_PATH")" "$(dirname "$ERROR_LOG")"
    ulimit -n "$WORKER_RLIMIT_NOFILE" 2>/dev/null || true
    rm -f "$SOCKET_PATH"
    write_config
    "$NGINX_BIN" -c "$CONF_PATH"
    echo "nginx proxy listening on unix:${SOCKET_PATH}"
    echo "sglang upstreams: ${UPSTREAMS}"
}

run_foreground() {
    require_nginx

    if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
        echo "nginx proxy is already running with pid $(cat "$PID_PATH")" >&2
        exit 1
    fi

    mkdir -p "$(dirname "$SOCKET_PATH")" "$(dirname "$CONF_PATH")" "$(dirname "$PID_PATH")" "$(dirname "$ERROR_LOG")"
    ulimit -n "$WORKER_RLIMIT_NOFILE" 2>/dev/null || true
    rm -f "$SOCKET_PATH"
    write_config
    exec "$NGINX_BIN" -c "$CONF_PATH" -g "daemon off;"
}

stop() {
    if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
        require_nginx
        "$NGINX_BIN" -c "$CONF_PATH" -s stop
        rm -f "$SOCKET_PATH"
        echo "nginx proxy stopped"
    else
        echo "nginx proxy is not running"
    fi
}

status() {
    if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
        echo "nginx proxy running with pid $(cat "$PID_PATH")"
        echo "socket: ${SOCKET_PATH}"
        echo "upstreams: ${UPSTREAMS}"
    else
        echo "nginx proxy is not running"
    fi
}

case "${1:-start}" in
    start)
        start
        ;;
    run)
        run_foreground
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        start
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|run|stop|restart|status}" >&2
        exit 2
        ;;
esac
