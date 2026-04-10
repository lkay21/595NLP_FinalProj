#!/bin/bash
set -e

for n in 6 8; do
    echo "=== Running with max-people = $n ==="
    python limem_demo.py full --max-people $n --save results_${n}.csv
done