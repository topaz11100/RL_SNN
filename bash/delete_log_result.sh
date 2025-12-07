#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results"

find "${PROJECT_ROOT}/logs" -mindepth 1 -delete
find "${PROJECT_ROOT}/results" -mindepth 1 -delete