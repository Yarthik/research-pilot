#!/usr/bin/env bash
# Push this repository to a Hugging Face Docker Space (works in Git Bash on Windows too).
#
#   export HF_TOKEN=hf_xxx            # a token with write access
#   ./scripts/deploy_hf_space.sh <hf-username>/<space-name>
#
#   DRY_RUN=1 ./scripts/deploy_hf_space.sh <hf-username>/<space-name>   # build the upload, don't push
#
# Deploys the last commit (uncommitted changes are not included). The Space gets its own README
# (with the YAML header Spaces needs); everything else is the repo. Set the Space's variables and
# secrets in its Settings (see docs/DEPLOY.md).
set -euo pipefail

SPACE="${1:?usage: deploy_hf_space.sh <hf-username>/<space-name>}"
DRY_RUN="${DRY_RUN:-}"
if [ -z "$DRY_RUN" ]; then
  : "${HF_TOKEN:?set HF_TOKEN to a Hugging Face token with write access}"
fi
HF_USER="${SPACE%%/*}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# core.autocrlf=false: export the committed bytes as they are. With autocrlf=true (the Git for
# Windows default) `git archive` would write CRLF line endings into the files.
git -C "$ROOT" -c core.autocrlf=false archive HEAD | tar -x -C "$WORK"
git -C "$ROOT" -c core.autocrlf=false show HEAD:deploy/huggingface/README_SPACE.md > "$WORK/README.md"

cd "$WORK"
git init -q -b main
git -c core.autocrlf=false add -A
git -c core.autocrlf=false -c user.name="deploy" -c user.email="deploy@localhost" commit -qm "Deploy ResearchPilot"
echo "Prepared $(git ls-files | wc -l | tr -d ' ') files from commit $(git -C "$ROOT" rev-parse --short HEAD)."

if [ -n "$DRY_RUN" ]; then
  echo "DRY_RUN set: not pushing to https://huggingface.co/spaces/${SPACE}"
  exit 0
fi
git push --force "https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${SPACE}" main
echo "Pushed. The Space builds at https://huggingface.co/spaces/${SPACE}"
