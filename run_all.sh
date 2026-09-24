#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${Q1_PYTHON:-}" ]]; then
  task_python="$Q1_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  task_python=.venv/bin/python
else
  task_python=python
fi
"$task_python" run.py "$@"
