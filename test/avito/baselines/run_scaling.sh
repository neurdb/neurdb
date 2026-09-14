#!/usr/bin/env bash
# ============================================================================
# run_scaling.sh -- data-scaling sweep: NeurEngine vs LOTUS vs Palimpzest.
# ============================================================================
# Runs the full AdCTR NLQ on 4 database scales (see build_scaled_dbs.sh):
#
#   scale    db           searchstream rows
#   mini     avito_mini    0.93M  (0.1x)
#   small    avito_small   4.6M   (0.5x)
#   medium   avito         9.25M  (1x, the original)
#   large    avito_large   37M    (4x)
#
# Per scale: NeurEngine in-database run (reuse + sched on, single AI server),
# then parquet export, then LOTUS and Palimpzest on the exported data.
# Baseline totals include their export time (it is part of the
# export-execute-import pattern); NeurEngine has no export.
#
# Prereqs: bash ../workloads/build_scaled_dbs.sh ; conda envs bl_lotus / bl_pz.
# Usage:   bash run_scaling.sh [mini small medium large]
# Output:  logs/scaling/<scale>_<system>.log, logs/scaling/scaling_results.csv
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORKLOADS="$HERE/../workloads"
LOGS="$HERE/logs/scaling"
mkdir -p "$LOGS"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

DEVICE="${DEVICE:-cuda:0}"
SCALES=("$@")
[ ${#SCALES[@]} -eq 0 ] && SCALES=(mini small medium large)

db_of() { case $1 in mini) echo avito_mini;; small) echo avito_small;;
                     medium) echo avito;; large) echo avito_large;; esac; }

# ---- warmup: absorb AI-server cold start before any timed run ---------------
PSQL="/code/neurdb-dev/build/psql/bin/psql -h 0.0.0.0 -U neurdb -d avito -v ON_ERROR_STOP=1"
WARMUP_SQL="SET nr_task_batch_size TO 512; SET nr_task_num_batches TO 1; SET nr_task_epoch TO 1; \
SELECT count(*) FROM (PREDICT VALUE OF label_ctr FROM (SELECT * FROM w_task_1 LIMIT 512) AS t TRAIN tabpfn ON *) AS p;"
echo "== warmup (excluded from timing) =="
docker exec neurdb_dev bash -lc "$PSQL -q -c \"$WARMUP_SQL\"" > /dev/null
docker exec neurdb_dev bash -lc "$PSQL -q -c \"$WARMUP_SQL\"" > /dev/null

for scale in "${SCALES[@]}"; do
  db=$(db_of "$scale")
  echo "==================== scale=$scale (db=$db) ===================="

  echo ">> NeurEngine ($db)"
  DB=$db CACHE=on MODES=on bash "$WORKLOADS/run_tasks.sh" 2>&1 | tee "$LOGS/${scale}_neurengine.log" | grep '^TIMING,' || true

  echo ">> export $db -> data_$scale/"
  conda activate bl_lotus
  PGDATABASE=$db python "$HERE/export_data.py" --out "$HERE/data_$scale" 2>&1 | tee "$LOGS/${scale}_export.log"

  echo ">> LOTUS ($scale)"
  python "$HERE/run_lotus.py" --data "$HERE/data_$scale" --out "$LOGS/lotus_$scale" --device "$DEVICE" 2>&1 \
    | tee "$LOGS/${scale}_lotus.log" | grep '^TIMING,' || true
  conda deactivate

  echo ">> Palimpzest ($scale)"
  conda activate bl_pz
  python "$HERE/run_palimpzest.py" --data "$HERE/data_$scale" --out "$LOGS/pz_$scale" --device "$DEVICE" 2>&1 \
    | tee "$LOGS/${scale}_palimpzest.log" | grep '^TIMING,' || true
  conda deactivate
done

# ---- aggregate: one row per (scale, system) with total seconds --------------
python3 - "$LOGS" <<'EOF'
import csv, os, re, sys

logs = sys.argv[1]
rows = []
for f in sorted(os.listdir(logs)):
    m = re.match(r"(\w+)_(neurengine|lotus|palimpzest|export)\.log$", f)
    if not m:
        continue
    scale, system = m.groups()
    t = {}
    for line in open(os.path.join(logs, f)):
        mm = re.match(r"TIMING,([^,]+),([\d.]+)", line.strip())
        if mm:
            t[mm.group(1)] = float(mm.group(2))
    if system == "neurengine":
        total = sum(t.values())
    elif system == "export":
        total = t.get("export_total", 0.0)
    else:
        total = t.get("total", 0.0)
    rows.append({"scale": scale, "system": system, "seconds": round(total, 1)})

out = os.path.join(logs, "scaling_results.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["scale", "system", "seconds"])
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out}")
for r in rows:
    print(r)
EOF

echo "== done =="
