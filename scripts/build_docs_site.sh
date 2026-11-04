#!/usr/bin/env bash
# Build the docs of every branch on origin into one site for GitHub Pages:
#
#   <out>/<slug>/       one MkDocs build per branch ("/" in names becomes "-")
#   <out>/versions.json the branch list read by Material's header selector
#   <out>/index.html    redirect to main/
#   <out>/404.html      redirect pre-branch links (/survpower/cli/) to main/
#
# main is built with --strict and must succeed; a branch whose docs fail to
# build is skipped with a warning. Usage:
#
#   scripts/build_docs_site.sh [out_dir]       # default: site
#
# DOCS_BASE_URL overrides the published base URL.
set -euo pipefail

OUT=${1:-site}
BASE=${DOCS_BASE_URL:-https://joansano.github.io/survpower}
BASE=${BASE%/}
REPO=$(git rev-parse --show-toplevel)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"; git -C "$REPO" worktree prune' EXIT

rm -rf "$OUT" && mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd)
git -C "$REPO" fetch --quiet --prune origin '+refs/heads/*:refs/remotes/origin/*'

built=()
for branch in $(git -C "$REPO" ls-remote --heads origin | sed 's#.*refs/heads/##'); do
  slug=${branch//\//-}
  wt="$WORK/$slug"
  git -C "$REPO" worktree add --quiet --detach "$wt" "origin/$branch"
  strict=()
  [ "$branch" = main ] && strict=(--strict)
  echo "building docs for $branch -> $slug/"
  if DOCS_SITE_URL="$BASE/$slug/" mkdocs build --quiet ${strict[@]+"${strict[@]}"} \
       -f "$wt/mkdocs.yml" -d "$OUT/$slug"; then
    built+=("$slug=$branch")
  elif [ "$branch" = main ]; then
    echo "error: docs for main failed to build" >&2
    exit 1
  else
    echo "warning: docs for branch '$branch' failed to build; skipped" >&2
    rm -rf "${OUT:?}/$slug"
  fi
  git -C "$REPO" worktree remove --force "$wt"
done

python3 - "$OUT" "$BASE" ${built[@]+"${built[@]}"} <<'EOF'
import json, sys
from pathlib import Path
from urllib.parse import urlparse

out, base, pairs = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
versions = [dict(zip(("version", "title"), p.split("=", 1)), aliases=[]) for p in pairs]
versions.sort(key=lambda v: (v["version"] != "main", v["version"]))
(out / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")

prefix = urlparse(base).path.rstrip("/") + "/"
(out / "index.html").write_text(
    '<!doctype html><meta charset="utf-8"><title>survpower</title>\n'
    '<meta http-equiv="refresh" content="0; url=main/">\n'
    '<link rel="canonical" href="main/"><a href="main/">survpower documentation</a>\n')
(out / "404.html").write_text(f"""<!doctype html>
<meta charset="utf-8"><title>Page not found - survpower</title>
<p>Page not found. <a href="{prefix}main/">Go to the survpower documentation.</a></p>
<script>
  // links from before branch docs (/survpower/cli/) now live under main/
  var prefix = {json.dumps(prefix)}, versions = {json.dumps([v["version"] for v in versions])};
  var path = location.pathname;
  if (path.indexOf(prefix) === 0) {{
    var rest = path.slice(prefix.length);
    if (versions.indexOf(rest.split("/")[0]) === -1)
      location.replace(prefix + "main/" + rest + location.search + location.hash);
  }}
</script>
""")
print("versions:", ", ".join(v["title"] for v in versions))
EOF
