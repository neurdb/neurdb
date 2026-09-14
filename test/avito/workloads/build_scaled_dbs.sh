#!/usr/bin/env bash
# ============================================================================
# build_scaled_dbs.sh -- scaled copies of the avito DB for the data-scaling
#                        experiment (mini 0.1x / small 0.5x / [avito 1x] / large 4x)
# ============================================================================
# Each scale is a full TEMPLATE copy of avito (schema + indexes + AI engine
# registration), then the EVENT STREAMS (searchstream / visitstream /
# phonerequestsstream) are scaled. On downscaled copies the ad catalog
# (adsinfo) is pruned to the ads still referenced by some event, so each scale
# is a coherent subsample; category/location stay untouched. Upscaling
# duplicates events only (same catalog, more traffic).
#
#   * downscale: deterministic row sample (setseed + random() < f), VACUUM FULL
#   * upscale:   self-duplication (INSERT ... SELECT, applied twice -> 4x) with
#                indexes dropped/recreated around the inserts
#
# Timestamps are preserved, so the cutoff grid and label/feature semantics are
# unchanged; only the data volume the relational pipeline must chew through
# scales. Safe to re-run: existing scale DBs are dropped and rebuilt.
#
# Usage:  bash build_scaled_dbs.sh [mini small large]
# ============================================================================
set -euo pipefail

CONTAINER=neurdb_dev
PSQL="/code/neurdb-dev/build/psql/bin/psql -h 0.0.0.0 -U neurdb -v ON_ERROR_STOP=1"

run() { docker exec "$CONTAINER" bash -lc "$PSQL $*"; }

STREAMS="searchstream visitstream phonerequestsstream"

make_db() {  # make_db <name>
  local name=$1
  echo ">> creating $name (template copy of avito)"
  run "-d neurdb -c \"DROP DATABASE IF EXISTS $name;\""
  run "-d neurdb -c \"CREATE DATABASE $name TEMPLATE avito;\""
}

downscale() {  # downscale <name> <fraction>
  local name=$1 frac=$2
  for t in $STREAMS; do
    echo ">> $name: sampling $t to $frac"
    run "-d $name -c \"SELECT setseed(0.42); DELETE FROM $t WHERE random() >= $frac;\""
    run "-d $name -c \"VACUUM FULL ANALYZE $t;\""
  done
  # keep the catalog coherent with the sampled events: only ads that still
  # occur in some stream survive (referential integrity preserved; labels and
  # features are unchanged since label ads always have events).
  echo ">> $name: pruning adsinfo to referenced ads"
  run "-d $name -c \"SET work_mem='512MB'; SET max_parallel_workers_per_gather=8;
    DELETE FROM adsinfo a WHERE NOT EXISTS (SELECT 1 FROM searchstream s WHERE s.adid = a.adid)
      AND NOT EXISTS (SELECT 1 FROM visitstream v WHERE v.adid = a.adid)
      AND NOT EXISTS (SELECT 1 FROM phonerequestsstream p WHERE p.adid = a.adid);\""
  run "-d $name -c \"VACUUM FULL ANALYZE adsinfo;\""
}

upscale_4x() {  # upscale_4x <name>
  local name=$1
  for t in $STREAMS; do
    echo ">> $name: duplicating $t to 4x (indexes dropped during load)"
    local idx
    idx=$(run "-d $name -t -A -c \"SELECT indexname FROM pg_indexes WHERE tablename='$t' AND indexname NOT LIKE '%pkey'\"")
    for i in $idx; do run "-d $name -c \"DROP INDEX $i;\""; done
    run "-d $name -c \"INSERT INTO $t SELECT * FROM $t;\""
    run "-d $name -c \"INSERT INTO $t SELECT * FROM $t;\""
  done
  echo ">> $name: recreating indexes"
  run "-d $name -c \"SET maintenance_work_mem='1GB';
    CREATE INDEX idx_searchstream_searchdate ON searchstream(searchdate);
    CREATE INDEX idx_searchstream_adid       ON searchstream(adid);
    CREATE INDEX idx_searchstream_searchid   ON searchstream(searchid);
    CREATE INDEX idx_visitstream_viewdate    ON visitstream(viewdate);
    CREATE INDEX idx_visitstream_adid        ON visitstream(adid);
    CREATE INDEX idx_phonerequestsstream_phonerequestdate ON phonerequestsstream(phonerequestdate);
    CREATE INDEX idx_phonerequestsstream_adid ON phonerequestsstream(adid);\""
  run "-d $name -c \"VACUUM ANALYZE;\""
}

summary() {  # summary <name>
  run "-d $1 -c \"SELECT 'searchstream' AS t, count(*) FROM searchstream
    UNION ALL SELECT 'visitstream', count(*) FROM visitstream
    UNION ALL SELECT 'phonerequestsstream', count(*) FROM phonerequestsstream;
    SELECT pg_size_pretty(pg_database_size(current_database())) AS db_size;\""
}

SCALES=("$@")
[ ${#SCALES[@]} -eq 0 ] && SCALES=(mini small large)
for scale in "${SCALES[@]}"; do
  case $scale in
    mini)  make_db avito_mini;  downscale avito_mini 0.1;  summary avito_mini ;;
    small) make_db avito_small; downscale avito_small 0.5; summary avito_small ;;
    large) make_db avito_large; upscale_4x avito_large;    summary avito_large ;;
    *) echo "unknown scale: $scale" >&2; exit 1 ;;
  esac
done
echo "== done =="
