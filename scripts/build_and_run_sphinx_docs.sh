#!/usr/bin/env bash
# Build and optionally serve the PyPOTS Sphinx documentation.
#
# Usage (from the repository root; the script also works from other directories):
#   bash scripts/build_and_run_sphinx_docs.sh                       # Incremental build and preview
#   bash scripts/build_and_run_sphinx_docs.sh --clean               # Full rebuild and preview
#   bash scripts/build_and_run_sphinx_docs.sh --jobs 4              # Four parallel build jobs
#   bash scripts/build_and_run_sphinx_docs.sh --jobs auto           # Sphinx chooses the job count
#   bash scripts/build_and_run_sphinx_docs.sh --port 8009           # Preview on another port
#   bash scripts/build_and_run_sphinx_docs.sh --build-only          # Build without serving
#   bash scripts/build_and_run_sphinx_docs.sh --clean --strict --build-only
#                                                                 # Full build, warnings are errors
#   bash scripts/build_and_run_sphinx_docs.sh --help                # Show all options
#
# Use the Python environment with PyPOTS and its Sphinx dependencies installed.
# By default, cached doctrees are reused and the preview is served only on
# http://127.0.0.1:8008. Press Ctrl+C to stop the preview server.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="${ROOT_DIR}/docs"
BUILD_DIR="${DOCS_DIR}/_build"

PORT=8008
JOBS=1
CLEAN=false
STRICT=false
SERVE=true

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --clean        Rebuild documentation from scratch
  --port PORT    HTTP server port (default: 8008)
  --jobs N       Parallel build jobs (default: 1, or auto)
  --strict       Treat Sphinx warnings as errors
  --build-only   Build without starting the HTTP server
  -h, --help     Show this help message
EOF
}

fail() {
    printf "Error: %s\n" "$*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=true
            shift
            ;;
        --port)
            [[ $# -ge 2 ]] || fail "--port requires a port number"
            PORT="$2"
            [[ "$PORT" =~ ^[0-9]{1,5}$ ]] || fail "Port must be an integer from 1 to 65535"
            PORT=$((10#$PORT))
            (( PORT >= 1 && PORT <= 65535 )) || fail "Port must be an integer from 1 to 65535"
            shift 2
            ;;
        --jobs)
            [[ $# -ge 2 ]] || fail "--jobs requires a positive integer or auto"
            JOBS="$2"
            [[ "$JOBS" == auto || "$JOBS" =~ ^[1-9][0-9]*$ ]] || fail "Jobs must be a positive integer or auto"
            shift 2
            ;;
        --strict)
            STRICT=true
            shift
            ;;
        --build-only)
            SERVE=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown option: $1 (see --help)"
            ;;
    esac
done

command -v python >/dev/null 2>&1 || fail "Python is not available in the current environment"
python -m sphinx --version >/dev/null || fail "Sphinx is not available in the current Python environment"

if [[ "$CLEAN" == true ]]; then
    echo "Cleaning previous documentation build..."
    rm -rf -- "$BUILD_DIR"
fi

SPHINX_ARGS=(-j "$JOBS")
if [[ "$STRICT" == true ]]; then
    SPHINX_ARGS+=(-W --keep-going)
fi

echo "Building PyPOTS documentation..."

python -m sphinx \
    -b html \
    -d "${BUILD_DIR}/doctrees" \
    "${SPHINX_ARGS[@]}" \
    "$DOCS_DIR" \
    "${BUILD_DIR}/html"

echo "Documentation built successfully."

if [[ "$SERVE" == true ]]; then
    echo "Documentation: http://127.0.0.1:${PORT}"
    echo "Press Ctrl+C to stop the server."

    exec python -m http.server "$PORT" \
        --bind 127.0.0.1 \
        --directory "${BUILD_DIR}/html"
fi
