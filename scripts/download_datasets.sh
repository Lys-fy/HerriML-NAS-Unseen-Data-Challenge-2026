#!/usr/bin/env bash
#
# Download the public NAS Unseen Data Challenge datasets from figshare.
#
# Idempotent: re-running skips files whose local size matches the remote size,
# and resumes-by-redownload anything that doesn't match.
#
# Defaults to writing into <repo>/datasets/. Override with $DATASETS_DIR.
# Pure bash + curl + python3 — no jq dependency.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="${DATASETS_DIR:-$SCRIPT_DIR/../datasets}"

# Smallest first so partial / interrupted runs are still useful.
# Format: "name figshare_article_id"
DATASETS=(
    "Sudoku 26976121"
    "GameOfLife 30000835"
    "Gutenberg 24574753"
    "Language 24574729"
    "Chesseract 24118743"
    "AddNIST 24574354"
    "MultNIST 24574678"
    "Voxel 26970223"
    "GeoClassing 24050256"
    "CIFARTile 24551539"
    "Myofibre 26969998"
)

# Total ~20 GB. Cryptic and Windspeed are not included — the starter-kit DOIs
# (10.7488/ds/8053 and ds/8054) resolve to unrelated content; ask the organisers
# for correct links.

usage() {
    cat <<EOF
Usage: $0 [-h|--help] [DATASET ...]

Download public competition datasets from figshare into:
  $DATASETS_DIR

Without arguments, downloads all 11 datasets (~20 GB).
Pass dataset names as positional arguments to filter.

Datasets (smallest first):
$(printf '  %s\n' "${DATASETS[@]}" | awk '{print "  - "$1}')

Environment:
  DATASETS_DIR    Override the destination directory.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

filter=("$@")

contains() {
    local needle="$1"; shift
    for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
    return 1
}

mkdir -p "$DATASETS_DIR"

total_failures=0

for entry in "${DATASETS[@]}"; do
    read -r name figshare_id <<< "$entry"

    if [[ ${#filter[@]} -gt 0 ]] && ! contains "$name" "${filter[@]}"; then
        continue
    fi

    printf '\n=== %s (figshare %s) ===\n' "$name" "$figshare_id"
    dest="$DATASETS_DIR/$name"
    mkdir -p "$dest"

    if ! api_json=$(curl -sf "https://api.figshare.com/v2/articles/$figshare_id"); then
        printf '  ERROR: failed to fetch figshare metadata for %s\n' "$name" >&2
        total_failures=$((total_failures + 1))
        continue
    fi

    # Tab-separated lines: name <TAB> size <TAB> download_url
    files_tsv=$(echo "$api_json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for f in data.get('files', []):
    print(f\"{f['name']}\t{f['size']}\t{f['download_url']}\")
")

    while IFS=$'\t' read -r filename size url; do
        [[ -z "${filename:-}" ]] && continue
        target="$dest/$filename"

        if [[ -f "$target" ]]; then
            local_size=$(stat -c%s "$target" 2>/dev/null || echo 0)
            if [[ "$local_size" == "$size" ]]; then
                printf '  skip   %s (%s bytes, already complete)\n' "$filename" "$size"
                continue
            fi
            printf '  retry  %s (local %s != remote %s)\n' "$filename" "$local_size" "$size"
        fi

        printf '  fetch  %s (%s bytes)\n' "$filename" "$size"
        if ! curl -fL --progress-bar "$url" -o "$target.tmp"; then
            printf '  ERROR: failed to download %s\n' "$filename" >&2
            rm -f "$target.tmp"
            total_failures=$((total_failures + 1))
            continue
        fi
        mv "$target.tmp" "$target"
    done <<< "$files_tsv"
done

echo
echo "Done."
echo "Destination: $DATASETS_DIR"
echo
echo "Per-dataset disk usage:"
du -sh "$DATASETS_DIR"/*/ 2>/dev/null | sort -h || true

if [[ $total_failures -gt 0 ]]; then
    echo
    echo "WARNING: $total_failures file(s) failed to download. Re-run to retry." >&2
    exit 1
fi
