#!/usr/bin/env bash
# Download + extract the KGS dan-level SGF archives (u-go.net) for the Go (Sedol)
# dataset. Self-contained, idempotent (skips already-downloaded archives).
# URLs are in ~/local/go-sgf/all_urls.txt (scraped from the gamerecords index).
set -u
ROOT="$HOME/local/go-sgf"
cd "$ROOT" || exit 1
mkdir -p archives sgf
N=$(wc -l < all_urls.txt)
echo "downloading $N archives into $ROOT/archives ..."
i=0
while read -r url; do
  i=$((i+1))
  fn="archives/$(basename "$url")"
  if [ -s "$fn" ]; then echo "[$i/$N] have $(basename "$url")"; continue; fi
  if wget -q -O "$fn" "$url"; then echo "[$i/$N] got $(basename "$url")"; else echo "[$i/$N] FAIL $url"; rm -f "$fn"; fi
done < all_urls.txt
echo "extracting ..."
for a in archives/*.tar.gz; do tar -xzf "$a" -C sgf 2>/dev/null; done
echo "sgf files: $(find sgf -name '*.sgf' | wc -l)"
echo done
