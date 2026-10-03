#!/bin/bash
# Sequential runs (one model fits in 16 GB at a time). Resumable: re-run this script after an interruption.
cd "$(dirname "$0")" && source ../.venv/bin/activate
export PYTHONUNBUFFERED=1
for v in adjusted teacher hybrid report; do
  [ -f runs/$v.json ] && continue
  extra=""; [ $v = hybrid ] && extra="--attn_layers 5 10 15 20"
  python run_1b.py --variant $v --stage1 300 --stage2 150 $extra >> runs/$v.log 2>&1 || echo "$v failed"
  echo "$v done $(date)"
done
