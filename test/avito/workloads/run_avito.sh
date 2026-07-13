#!/usr/bin/env bash
# Build the avito AdCTR horizon-sweep tasks inside the neurdb_dev container,
# WITH the feature cache ON (reuse): cache is reset once, then every task only
# computes the features it doesn't already have.
# Usage:  bash run_avito.sh            (horizons 1 3 7)
#         HORIZONS="1 2 4" bash run_avito.sh
set -euo pipefail

CONTAINER=neurdb_dev
PSQL="/code/neurdb-dev/build/psql/bin/psql -h 0.0.0.0 -U neurdb -d avito -v ON_ERROR_STOP=1"
DIR=/code/neurdb-dev/test/avito/workloads
HORIZONS="${HORIZONS:-1 3 7}"

run() { docker exec "$CONTAINER" bash -lc "$PSQL $*"; }

echo ">> tool: cutoffs";       run "-f $DIR/tool_cutoffs.sql"
for h in $HORIZONS; do echo ">> 01 label h=$h"; run "-v h=$h -f $DIR/01_label_adctr.sql"; done
echo ">> tool: cache init (ONCE = reuse mode)"; run "-f $DIR/tool_feat_cache_init.sql"
for h in $HORIZONS; do
  echo ">> 02 features (delta) h=$h"; run "-v h=$h -f $DIR/02_features_adctr_pit.sql"
  echo ">> 03 task h=$h";             run "-v h=$h -f $DIR/03_task_table.sql"
done
echo ">> done"
