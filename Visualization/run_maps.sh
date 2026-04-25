#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/generate_city_comparison_maps.py"

echo
echo "Generated map files in:"
echo "  ${SCRIPT_DIR}/maps"
echo
echo "Open this interactive toggle map:"
echo "  ${SCRIPT_DIR}/maps/all_cities_postal_comparison_map.html"
