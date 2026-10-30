#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for attempt in $(seq 1 90); do
    [ -f results/edit-model.json ] && break
    sleep 10
done
test -f results/edit-model.json
EDIT_PYTHON=/home/ubuntu/cosmos-excavator-test/.venv/bin/python
EDIT_LIBS=$($EDIT_PYTHON -c 'import glob; print(":".join(sorted(glob.glob("/home/ubuntu/cosmos-excavator-test/.venv/lib/python3.13/site-packages/nvidia/*/lib"))))')
EDIT_GATEWAY=$(docker network inspect cdf-route1 --format '{{(index .IPAM.Config 0).Gateway}}')
export LD_LIBRARY_PATH="${EDIT_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false
exec "$EDIT_PYTHON" edit_server.py --host "$EDIT_GATEWAY" --port 8011
