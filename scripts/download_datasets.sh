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

# --- Edinburgh DataShare datasets (DSpace, NOT figshare) ------------------
# Windspeed / Cryptic: 2025 datasets, published late (2025-12-05) so the
# starter-kit DOIs (10.7488/ds/8053, ds/8054) now resolve. CC-BY-4.0.
# Each is a single <file>.tar.gz that unpacks to the standard competition
# layout (6 npy + metadata + README), possibly nested one dir deep.
# Format: "name dspace_handle_suffix tarfilename"
DATASHARE=(
    "Cryptic 9132 cryptic.tar.gz"
    "Windspeed 9131 windspeed.tar.gz"
)

for entry in "${DATASHARE[@]}"; do
    read -r name handle tarname <<< "$entry"

    if [[ ${#filter[@]} -gt 0 ]] && ! contains "$name" "${filter[@]}"; then
        continue
    fi

    printf '\n=== %s (Edinburgh DataShare 10283/%s) ===\n' "$name" "$handle"
    dest="$DATASETS_DIR/$name"

    if [[ -f "$dest/metadata" ]] && [[ -f "$dest/train_x.npy" ]]; then
        printf '  skip   %s (already extracted)\n' "$name"
        continue
    fi
    mkdir -p "$dest"

    url="https://datashare.ed.ac.uk/bitstream/handle/10283/$handle/$tarname"
    tmp="$dest/.$tarname.tmp"
    printf '  fetch  %s\n' "$url"
    if ! curl -fL -sS "$url" -o "$tmp"; then
        printf '  ERROR: download failed for %s\n' "$name" >&2
        rm -f "$tmp"; total_failures=$((total_failures + 1)); continue
    fi

    extract="$dest/.extract"
    rm -rf "$extract"; mkdir -p "$extract"
    if ! tar -xzf "$tmp" -C "$extract"; then
        printf '  ERROR: extract failed for %s\n' "$name" >&2
        rm -rf "$tmp" "$extract"; total_failures=$((total_failures + 1)); continue
    fi

    # Locate the dir actually containing the dataset (handles nested tars).
    meta=$(find "$extract" -maxdepth 4 -name metadata -type f 2>/dev/null | head -1 || true)
    if [[ -z "$meta" ]]; then
        printf '  ERROR: no metadata file found inside %s tarball\n' "$name" >&2
        rm -rf "$tmp" "$extract"; total_failures=$((total_failures + 1)); continue
    fi
    srcdir=$(dirname "$meta")
    shopt -s dotglob
    mv -f "$srcdir"/* "$dest"/
    shopt -u dotglob

    # Clean macOS AppleDouble sidecars and sanitise the metadata JSON: some
    # DataShare tars ship `"benchmark": ?` (an unfilled placeholder) which is
    # invalid JSON and would crash json.loads() in the harness. Normalise to
    # 0.0 — the same convention as Sudoku's shipped placeholder.
    find "$dest" -name '._*' -delete 2>/dev/null || true
    python3 - "$dest/metadata" <<'PYEOF' || true
import json, re, sys, pathlib
p = pathlib.Path(sys.argv[1])
try:
    txt = p.read_text()
    try:
        json.loads(txt)
    except Exception:
        md = json.loads(re.sub(r'("benchmark"\s*:\s*)\?', r'\g<1>0.0', txt))
        p.write_text(json.dumps(md))
        print(f"  note   sanitised metadata (placeholder benchmark -> 0.0)")
except Exception as e:
    print(f"  WARN   could not sanitise {p}: {e}")
PYEOF
    rm -rf "$tmp" "$extract"
    printf '  ok     %s extracted -> %s\n' "$name" "$dest"
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
