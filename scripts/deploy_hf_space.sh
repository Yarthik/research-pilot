#!/usr/bin/env bash
# Push this repository to a Hugging Face Docker Space.
#
#   export HF_TOKEN=hf_xxx            # a token with write access
#   ./scripts/deploy_hf_space.sh <hf-username>/<space-name>
#
# The Space gets its own README (with the YAML header Spaces needs); everything else is the repo.
# Set GROQ_API_KEY (and optionally ALLOW_INGEST=false) in the Space's Settings -> Variables and secrets.
set -euo pipefail

SPACE="${1:?usage: deploy_hf_space.sh <hf-username>/<space-name>}"
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face token with write access}"
HF_USER="${SPACE%%/*}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Deploys the last commit (uncommitted changes are not included).
git -C "$ROOT" archive HEAD | tar -x -C "$WORK"
cp "$ROOT/deploy/huggingface/README_SPACE.md" "$WORK/README.md"

cd "$WORK"
git init -q -b main
git add -A
git -c user.name="deploy" -c user.email="deploy@localhost" commit -qm "Deploy ResearchPilot"
git push --force "https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${SPACE}" main
echo "Pushed. The Space builds at https://huggingface.co/spaces/${SPACE}"
