#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

case "${1:-}" in
    course)
        shift
        if [[ -x .venv/bin/python ]]; then
            exec .venv/bin/python -m llm_lab dashboard --open "$@"
        fi
        exec python3 -m llm_lab dashboard --open "$@"
        ;;
    *)
        echo "Usage: $0 course [--host HOST] [--port PORT]" >&2
        exit 1
        ;;
esac
