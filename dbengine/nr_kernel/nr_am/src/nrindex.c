#include "nrindex.h"
#include "nrindex_access/nrindex_kv.h"
#include "postgres.h"
#include "access/amapi.h"
#include "access/reloptions.h"
#include "access/relscan.h"
#include "catalog/index.h"
#include "commands/vacuum.h"
#include "nodes/pathnodes.h"
#include "utils/guc.h"
#include "utils/rel.h"
#include "utils/memutils.h"
#include "utils/builtins.h"
#include "storage/ipc.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "fmgr.h"
#include "access/heapam.h"
#include "access/htup_details.h"
#include "executor/tuptable.h"
#include "utils/elog.h"
#include "utils/snapmgr.h"
#include "access/xact.h"
#include "access/heapam.h"
#include "access/multixact.h"
#include "nram_storage/rocks_service.h"
#include "nram_storage/rocks_handler.h"
#include "nram_storage/indexengine.h"  /* 直接调用 IndexEngine */
#include "nram_access/kv.h"
#include "nram_xact/xact.h"
#include "nram_xact/action.h"
#include "access/stratnum.h"  /* BTEqualStrategyNumber 等 */

// PG_MODULE_MAGIC;

/* ------------------------------------------------------------------------
 * Global variables and structures
 * ------------------------------------------------------------------------
 */

/* Index scan descriptor for nrindex - now using proper index structures */

/* Index build state for nrindex */
typedef struct NRIndexBuildState
{
    Relation heap;
    Relation index;
    IndexInfo *indexInfo;
    NRIndexKey **keys;
    NRIndexValue **values;
    int count;
    int capacity;
} NRIndexBuildState;

/* ------------------------------------------------------------------------
 * Helper functions
 * ------------------------------------------------------------------------
 */

/* Helper functions now use the new index key-value structures */

/* ------------------------------------------------------------------------
 * Index AM implementation functions
 * ------------------------------------------------------------------------
 */

/*
 * Build a new index using bulk load for better performance.
 */
static IndexBuildResult *
nrindex_build(Relation heap, Relation index, IndexInfo *indexInfo)
{
    IndexBuildResult *result;
    TableScanDesc scan;
    HeapTuple heapTuple;
    TupleDesc heapTupDesc;
    TupleDesc indexTupDesc;
    Datum values[INDEX_MAX_KEYS];
    bool isnull[INDEX_MAX_KEYS];
    int nkeys;
    int ntuples = 0;

    /* Bulk load data structures */
    int capacity = 100000;  /* Initial capacity */
    int64 *bulk_keys;
    uint64 *bulk_values;
    Oid key_type_oid;
    bool is_bigint;

    elog(LOG, "========== NRINDEX BUILD START (BULK LOAD) ==========");
    elog(LOG, "Building index on table: %s (OID: %u)",
         RelationGetRelationName(heap), heap->rd_id);
    elog(LOG, "Index name: %s (OID: %u)",
         RelationGetRelationName(index), index->rd_id);

    heapTupDesc = RelationGetDescr(heap);
    indexTupDesc = RelationGetDescr(index);
    nkeys = indexInfo->ii_NumIndexKeyAttrs;

    elog(LOG, "Number of index key columns: %d", nkeys);

    result = (IndexBuildResult *) palloc(sizeof(IndexBuildResult));

    /* Determine key type (INT or BIGINT) */
    {
        int heapAttrNum = indexInfo->ii_IndexAttrNumbers[0];
        Form_pg_attribute attr = TupleDescAttr(heapTupDesc, heapAttrNum - 1);
        key_type_oid = attr->atttypid;
        is_bigint = (key_type_oid == INT8OID);
        elog(LOG, "Key column type: %s (OID: %u)", is_bigint ? "BIGINT" : "INT", key_type_oid);
    }

    /* Allocate bulk load arrays */
    bulk_keys = (int64 *) palloc(sizeof(int64) * capacity);
    bulk_values = (uint64 *) palloc(sizeof(uint64) * capacity);

    /* Phase 1: Scan heap and collect all key-value pairs */
    elog(LOG, "Phase 1: Scanning heap table and collecting data...");
    scan = table_beginscan(heap, SnapshotAny, 0, NULL);

    while ((heapTuple = heap_getnext(scan, ForwardScanDirection)) != NULL) {
        int heapAttrNum;
        int64 key_val;
        uint64 compressed_tid;
        BlockNumber blk;
        OffsetNumber off;

        /* Check capacity and expand if needed */
        if (ntuples >= capacity) {
            capacity *= 2;
            bulk_keys = (int64 *) repalloc(bulk_keys, sizeof(int64) * capacity);
            bulk_values = (uint64 *) repalloc(bulk_values, sizeof(uint64) * capacity);
            elog(LOG, "Expanded capacity to %d", capacity);
        }

        /* Extract index key value */
        heapAttrNum = indexInfo->ii_IndexAttrNumbers[0];
        values[0] = heap_getattr(heapTuple, heapAttrNum, heapTupDesc, &isnull[0]);

        if (isnull[0]) {
            /* Skip NULL values for now */
            continue;
        }

        /* Get key value (INT or BIGINT) */
        if (is_bigint) {
            key_val = DatumGetInt64(values[0]);
        } else {
            key_val = (int64)DatumGetInt32(values[0]);
        }

        /* Compress heap_tid to uint64: high 32 bits = block, low 16 bits = offset */
        blk = ItemPointerGetBlockNumber(&heapTuple->t_self);
        off = ItemPointerGetOffsetNumber(&heapTuple->t_self);
        compressed_tid = ((uint64)blk << 32) | (uint64)off;

        /* Store in bulk arrays */
        bulk_keys[ntuples] = key_val;
        bulk_values[ntuples] = compressed_tid;

        ntuples++;

        /* Progress logging every 100000 tuples */
        if (ntuples % 100000 == 0) {
            elog(LOG, "  Collected %d tuples...", ntuples);
        }
    }

    table_endscan(scan);

    elog(LOG, "Phase 1 complete: collected %d tuples", ntuples);

    /* Phase 2: Bulk load into index engine (direct call - no IPC) */
    if (ntuples > 0) {
        elog(LOG, "Phase 2: Bulk loading %d entries into index engine (direct call)...", ntuples);

        nrindex_rocks_bulk_load(index->rd_id, bulk_keys, bulk_values, ntuples);

        elog(LOG, "Phase 2 complete: bulk load successful");
    }

    /* Clean up */
    pfree(bulk_keys);
    pfree(bulk_values);

    result->heap_tuples = ntuples;
    result->index_tuples = ntuples;

    elog(LOG, "========== NRINDEX BUILD COMPLETE ==========");
    elog(LOG, "Total tuples indexed: %d", ntuples);

    return result;
}

/*
 * Build an empty index.
 */
static void
nrindex_buildempty(Relation index)
{
    /* No need to build an init fork for RocksDB-based index */
}

/*
 * Insert new tuple to index.
 */
static bool
nrindex_insert(Relation index, Datum *values, bool *isnull,
               ItemPointer ht_ctid, Relation heapRel,
               IndexUniqueCheck checkUnique,
               bool indexUnchanged,
               IndexInfo *indexInfo)
{
    NRIndexKey ikey;
    NRIndexValue ivalue;
    bool result = true;

    elog(DEBUG1, ">>> NRINDEX INSERT <<<");
    elog(DEBUG1, "Index: %s (OID: %u), inserting for ctid=(%u,%u)",
         RelationGetRelationName(index), index->rd_id,
         ItemPointerGetBlockNumber(ht_ctid),
         ItemPointerGetOffsetNumber(ht_ctid));

    /* Build index key and value using new structures */
    ikey = nrindex_key_create(index->rd_id, values, isnull, indexInfo->ii_NumIndexKeyAttrs, RelationGetDescr(index));
    ivalue = nrindex_value_create(ht_ctid);

    elog(DEBUG1, "Created key: indexOid=%u, key_size=%u",
         ikey->indexOid, ikey->key_size);

    /* Check for uniqueness if required */
    if (checkUnique != UNIQUE_CHECK_NO) {
        elog(DEBUG1, "Checking uniqueness...");
        NRIndexValue existing_value = nrindex_rocks_get(ikey);
        if (existing_value != NULL) {
            /* Check if it's the same tuple */
            if (!ItemPointerEquals(ht_ctid, &existing_value->heap_tid)) {
                elog(DEBUG1, "Duplicate key found!");
                result = false; /* Duplicate key violation */
            }
            nrindex_value_free(existing_value);
        }
    }

    if (result) {
        if (!nrindex_rocks_put(ikey, ivalue)) {
            elog(ERROR, "Failed to insert index entry");
        }
        elog(DEBUG1, "Insert successful!");
    }

    nrindex_key_free(ikey);
    nrindex_value_free(ivalue);

    return result;
}

/*
 * Bulk deletion of index entries.
 */
static IndexBulkDeleteResult *
nrindex_bulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
                   IndexBulkDeleteCallback callback, void *callback_state)
{
    /* For now, return NULL indicating no work was done */
    /* TODO: Implement bulk delete using RocksDB range operations */
    return NULL;
}

/*
 * Post-VACUUM cleanup for index.
 */
static IndexBulkDeleteResult *
nrindex_vacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
    /* For now, return NULL indicating no work was done */
    /* TODO: Implement vacuum cleanup */
    return NULL;
}

/*
 * Estimate cost of using this index.
 */
static void
nrindex_costestimate(PlannerInfo *root, IndexPath *path, double loop_count,
                     Cost *indexStartupCost, Cost *indexTotalCost,
                     Selectivity *indexSelectivity, double *indexCorrelation,
                     double *indexPages)
{
    /* Simple cost estimation */
    *indexStartupCost = 1.0;
    *indexTotalCost = path->path.rows + 1.0;
    *indexSelectivity = 1.0;
    *indexCorrelation = 0.0;
    *indexPages = 1.0;
}

/*
 * Parse relation options for index.
 */
static bytea *
nrindex_options(Datum reloptions, bool validate)
{
    /* No special options for now */
    return NULL;
}

/*
 * Validate operator class for index.
 */
static bool
nrindex_validate(Oid opclassoid)
{
    /* Accept any operator class for now */
    return true;
}

/*
 * Begin scan of index.
 */
static IndexScanDesc
nrindex_beginscan(Relation r, int nkeys, int norderbys)
{
    NRIndexScanDesc scan;

    elog(DEBUG1, "========== NRINDEX BEGINSCAN ==========");
    elog(DEBUG1, "Index: %s (OID: %u)", RelationGetRelationName(r), r->rd_id);
    elog(DEBUG1, "Number of scan keys: %d", nkeys);
    elog(DEBUG1, "Number of order by keys: %d", norderbys);

    scan = (NRIndexScanDesc) RelationGetIndexScan(r, nkeys, norderbys);
    scan->min_key = NULL;
    scan->max_key = NULL;
    scan->results_key = NULL;
    scan->results = NULL;
    scan->result_count = 0;
    scan->cursor = 0;
    scan->is_range_scan = false;
    scan->is_null_scan = false;
    /* Point query optimization fields */
    scan->is_point_query = false;
    scan->point_key = NULL;
    scan->point_result = NULL;
    scan->point_found = false;
    scan->point_returned = false;

    elog(DEBUG1, "Scan descriptor initialized");

    return (IndexScanDesc) scan;
}

/*
 * Rescan index.
 */
static void
nrindex_rescan(IndexScanDesc scan, ScanKey scankey, int nscankeys,
               ScanKey orderbys, int norderbys)
{
    NRIndexScanDesc nrscan = (NRIndexScanDesc) scan;

    elog(DEBUG1, "========== NRINDEX RESCAN ==========");
    elog(DEBUG1, "Number of scan keys: %d", nscankeys);

    /* 打印每个 scankey 的信息 */
    for (int i = 0; i < nscankeys; i++) {
        elog(DEBUG1, "  ScanKey[%d]:", i);
        elog(DEBUG1, "    sk_attno = %d (which column)", scankey[i].sk_attno);
        elog(DEBUG1, "    sk_strategy = %d (1:<, 2:<=, 3:=, 4:>=, 5:>)",
             scankey[i].sk_strategy);
        elog(DEBUG1, "    sk_flags = %d", scankey[i].sk_flags);
        /* 尝试打印查询值（假设是 int4） */
        if (!(scankey[i].sk_flags & SK_ISNULL)) {
            elog(DEBUG1, "    sk_argument = %d (search value)",
                 DatumGetInt32(scankey[i].sk_argument));
        } else {
            elog(DEBUG1, "    sk_argument = NULL");
        }
    }

    /* Free previous results */
    if (nrscan->results_key) {
        for (int i = 0; i < nrscan->result_count; i++) {
            nrindex_key_free(nrscan->results_key[i]);
            nrindex_value_free(nrscan->results[i]);
        }
        pfree(nrscan->results_key);
        pfree(nrscan->results);
    }

    nrscan->result_count = 0;
    nrscan->cursor = 0;

    /* Build scan keys for RocksDB range scan */
    /* -- 触发这个条件的 SQL SELECT * FROM test_idx WHERE val IS NULL; 
       - nscankeys > 0 — 有查询条件
       - scankey[0].sk_flags & SK_ISNULL — 位运算，检查是否设置了 SK_ISNULL 标志
    */
    if (nscankeys > 0 && scankey[0].sk_flags & SK_ISNULL) {
        /* Handle IS NULL scan */
        elog(DEBUG1, "Scan type: IS NULL scan");
        nrscan->is_range_scan = false;
        nrscan->is_null_scan = true;
        /* TODO: Implement NULL handling */
    } else if (nscankeys > 0) {
        /*  有扫描条件，且不是 IS NULL 查询（正常的值比较）
            情况 2: WHERE val = 500, val > 100, val < 1000 等
            正常的值比较查询
        */
        Relation indexRel = scan->indexRelation;
        TupleDesc indexTupDesc = RelationGetDescr(indexRel);
        Oid indexOid = RelationGetRelid(indexRel);

        /* 准备 values 和 isnull 数组 */
        Datum *values = palloc(sizeof(Datum) * nscankeys);
        bool *isnull = palloc(sizeof(bool) * nscankeys);

        for (int i = 0; i < nscankeys; i++) {
            if (scankey[i].sk_flags & SK_ISNULL) {
                isnull[i] = true;
                values[i] = (Datum) 0;
            } else {
                isnull[i] = false;
                values[i] = scankey[i].sk_argument;  /* 取出查询值 (如 500) */
            }
        }

        elog(DEBUG1, "Building search key: indexOid=%u, nscankeys=%d", indexOid, nscankeys);

        /* 根据操作符类型构建 key
           BTEqualStrategyNumber=3, BTLessStrategyNumber=1, BTLessEqualStrategyNumber=2,
           BTGreaterEqualStrategyNumber=4, BTGreaterStrategyNumber=5 */
        switch (scankey[0].sk_strategy) {
            case BTEqualStrategyNumber:  /* = 等值查询 */
                elog(DEBUG1, "Strategy: EQUAL (=) - using point lookup optimization");
                /* Use optimized point lookup instead of range scan */
                nrscan->is_point_query = true;
                nrscan->point_key = nrindex_key_create(indexOid, values, isnull,
                                                        nscankeys, indexTupDesc);
                nrscan->point_returned = false;
                break;

            case BTLessStrategyNumber:      /* < */
            case BTLessEqualStrategyNumber: /* <= */
                elog(DEBUG1, "Strategy: LESS (<) or LESS_EQUAL (<=)");
                /* min_key = NULL (从头开始), max_key = 查询值 */
                nrscan->min_key = NULL;
                nrscan->max_key = nrindex_key_create(indexOid, values, isnull,
                                                      nscankeys, indexTupDesc);
                break;

            case BTGreaterStrategyNumber:      /* > */
            case BTGreaterEqualStrategyNumber: /* >= */
                elog(DEBUG1, "Strategy: GREATER (>) or GREATER_EQUAL (>=)");
                /* min_key = 查询值, max_key = NULL (到结尾) */
                nrscan->min_key = nrindex_key_create(indexOid, values, isnull,
                                                      nscankeys, indexTupDesc);
                nrscan->max_key = NULL;
                break;

            default:
                elog(DEBUG1, "Unknown strategy: %d", scankey[0].sk_strategy);
                nrscan->min_key = NULL;
                nrscan->max_key = NULL;
                break;
        }

        pfree(values);
        pfree(isnull);

        nrscan->is_null_scan = false;

        if (nrscan->is_point_query) {
            /* Optimized path: use point lookup for equality queries */
            elog(DEBUG1, "Using point lookup optimization");

            nrindex_rocks_point_lookup(nrscan->point_key,
                                       &nrscan->point_result,
                                       &nrscan->point_found);

            elog(DEBUG1, "Point lookup found=%d", nrscan->point_found);
            nrscan->is_range_scan = false;
        } else {
            /* Range scan path for <, <=, >, >= operators */
            nrscan->is_range_scan = true;
            nrscan->is_point_query = false;

            elog(DEBUG1, "Calling nrindex_rocks_range_scan: min_key=%s, max_key=%s",
                 nrscan->min_key ? "SET" : "NULL",
                 nrscan->max_key ? "SET" : "NULL");

            /* Perform range scan using new index functions */
            if (!nrindex_rocks_range_scan(nrscan->min_key, nrscan->max_key,
                                         &nrscan->results_key, &nrscan->results,
                                         &nrscan->result_count)) {
                elog(DEBUG1, "Range scan returned no results or failed");
            }
            elog(DEBUG1, "Range scan result_count = %d", nrscan->result_count);
        }
    } else {
        /*
            情况 3: nscankeys == 0
            没有 WHERE 条件，全表扫描
            SELECT * FROM test_idx;
        */
        elog(DEBUG1, "No scan keys provided - full scan");
    }

    elog(DEBUG1, "========== NRINDEX RESCAN END ==========");
}

/*
 * Get next tuple from index scan.
 */
static bool
nrindex_gettuple(IndexScanDesc scan, ScanDirection direction)
{
    NRIndexScanDesc nrscan = (NRIndexScanDesc) scan;

    elog(DEBUG1, ">>> NRINDEX GETTUPLE <<<");

    if (direction != ForwardScanDirection) {
        elog(WARNING, "nrindex only supports forward scan");
        return false;
    }

    /* Handle point query (equality lookup) - optimized path */
    if (nrscan->is_point_query) {
        elog(DEBUG1, "Point query path: found=%d, returned=%d",
             nrscan->point_found, nrscan->point_returned);

        if (!nrscan->point_found || nrscan->point_returned) {
            elog(DEBUG1, "Point query: no more results");
            return false;
        }

        /* Return the single result */
        scan->xs_heaptid = nrscan->point_result->heap_tid;

        elog(DEBUG1, "Point query found: heap_tid=(%u,%u)",
             ItemPointerGetBlockNumber(&nrscan->point_result->heap_tid),
             ItemPointerGetOffsetNumber(&nrscan->point_result->heap_tid));

        nrscan->point_returned = true;
        return true;
    }

    /* Handle range scan - original path */
    elog(DEBUG1, "Range scan path: cursor=%d, result_count=%d",
         nrscan->cursor, nrscan->result_count);

    if (nrscan->cursor >= nrscan->result_count) {
        elog(DEBUG1, "No more results (cursor >= result_count)");
        return false;
    }

    /* Get the current result */
    NRIndexKey ikey = nrscan->results_key[nrscan->cursor];
    NRIndexValue ivalue = nrscan->results[nrscan->cursor];

    /* Set the tuple identifier */
    scan->xs_heaptid = ivalue->heap_tid;

    elog(DEBUG1, "Found result[%d]: heap_tid=(%u,%u)",
         nrscan->cursor,
         ItemPointerGetBlockNumber(&ivalue->heap_tid),
         ItemPointerGetOffsetNumber(&ivalue->heap_tid));

    nrscan->cursor++;

    return true;
}

/*
 * Get bitmap of matching tuples.
 */
static int64
nrindex_getbitmap(IndexScanDesc scan, TIDBitmap *tbm)
{
    NRIndexScanDesc nrscan = (NRIndexScanDesc) scan;
    int64 ntids = 0;

    /* Handle point query (equality lookup) - optimized path */
    if (nrscan->is_point_query) {
        if (nrscan->point_found && nrscan->point_result != NULL) {
            tbm_add_tuples(tbm, &nrscan->point_result->heap_tid, 1, false);
            ntids = 1;
        }
        return ntids;
    }

    /* Add all matching TIDs to bitmap - range scan path */
    for (int i = 0; i < nrscan->result_count; i++) {
        NRIndexValue ivalue = nrscan->results[i];

        tbm_add_tuples(tbm, &ivalue->heap_tid, 1, false);
        ntids++;
    }

    return ntids;
}

/*
 * End scan of index.
 */
static void
nrindex_endscan(IndexScanDesc scan)
{
    NRIndexScanDesc nrscan = (NRIndexScanDesc) scan;

    elog(DEBUG1, "========== NRINDEX ENDSCAN ==========");
    elog(DEBUG1, "Ending index scan, cleaning up resources");
    elog(DEBUG1, "  result_count = %d", nrscan->result_count);

    /* Free scan results */
    if (nrscan->results_key) {
        elog(DEBUG1, "  Freeing %d result key-value pairs", nrscan->result_count);
        for (int i = 0; i < nrscan->result_count; i++) {
            nrindex_key_free(nrscan->results_key[i]);
            nrindex_value_free(nrscan->results[i]);
        }
        pfree(nrscan->results_key);
        pfree(nrscan->results);
    }

    if (nrscan->min_key) {
        elog(DEBUG1, "  Freeing min_key");
        nrindex_key_free(nrscan->min_key);
    }
    if (nrscan->max_key) {
        elog(DEBUG1, "  Freeing max_key");
        nrindex_key_free(nrscan->max_key);
    }

    elog(DEBUG1, "========== NRINDEX ENDSCAN DONE ==========");
}

/*
 * Index AM handler function: returns IndexAmRoutine with access method
 * parameters and callbacks.
 */
Datum nrindex_handler(PG_FUNCTION_ARGS);
PG_FUNCTION_INFO_V1(nrindex_handler);

Datum
nrindex_handler(PG_FUNCTION_ARGS)
{
    IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);
    
    /* Set AM capabilities */
    amroutine->amstrategies = 0;  /* No fixed strategies */
    amroutine->amsupport = 1;     /* One support function */
    amroutine->amoptsprocnum = 0; /* No options procedure */
    amroutine->amcanorder = false;
    amroutine->amcanorderbyop = false;
    amroutine->amcanbackward = false;
    amroutine->amcanunique = true;
    amroutine->amcanmulticol = true;
    amroutine->amoptionalkey = true;
    amroutine->amsearcharray = false;
    amroutine->amsearchnulls = false;
    amroutine->amstorage = false;
    amroutine->amclusterable = false;
    amroutine->ampredlocks = false;
    amroutine->amcanparallel = false;
    amroutine->amcaninclude = false;
    amroutine->amusemaintenanceworkmem = false;
    amroutine->amsummarizing = false;
    amroutine->amparallelvacuumoptions = VACUUM_OPTION_NO_PARALLEL;
    amroutine->amkeytype = InvalidOid;
    
    /* Set function pointers */
    amroutine->ambuild = nrindex_build;
    amroutine->ambuildempty = nrindex_buildempty;
    amroutine->aminsert = nrindex_insert;
    amroutine->ambulkdelete = nrindex_bulkdelete;
    amroutine->amvacuumcleanup = nrindex_vacuumcleanup;
    amroutine->amcanreturn = NULL;
    amroutine->amcostestimate = nrindex_costestimate;
    amroutine->amoptions = nrindex_options;
    amroutine->amproperty = NULL;
    amroutine->ambuildphasename = NULL;
    amroutine->amvalidate = nrindex_validate;
    amroutine->amadjustmembers = NULL;
    amroutine->ambeginscan = nrindex_beginscan;
    amroutine->amrescan = nrindex_rescan;
    amroutine->amgettuple = nrindex_gettuple;
    amroutine->amgetbitmap = nrindex_getbitmap;
    amroutine->amendscan = nrindex_endscan;
    amroutine->ammarkpos = NULL;
    amroutine->amrestrpos = NULL;
    amroutine->amestimateparallelscan = NULL;
    amroutine->aminitparallelscan = NULL;
    amroutine->amparallelrescan = NULL;
    
    PG_RETURN_POINTER(amroutine);
}
