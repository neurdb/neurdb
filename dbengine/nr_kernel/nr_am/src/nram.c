#include "nram.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "utils/memutils.h"
#include "postgres.h"
#include "fmgr.h"
#include "access/tableam.h"
#include "access/heapam.h"
#include "access/xact.h"
#include "nodes/execnodes.h"
#include "catalog/index.h"
#include "commands/vacuum.h"
#include "utils/builtins.h"
#include "executor/tuptable.h"
#include "utils/elog.h"

PG_MODULE_MAGIC;

/* ------------------------------------------------------------------------
 * Helper functions.
 * ------------------------------------------------------------------------
 */

static KVEngine *current_session_engine = NULL;

void nram_shutdown_session(void) {
    NRAM_INFO();
    if (current_session_engine) {
        current_session_engine->destroy(current_session_engine);
        current_session_engine = NULL;
    }
}

List *nram_get_primary_key_attrs(Relation rel) {
    List *index_list = RelationGetIndexList(rel);
    ListCell *lc;
    List *key_attrs = NIL;

    NRAM_INFO();

    if (index_list == NIL) {
        // In Postgres, the tuple id is changed on every update,
        // making it an inefficient primary key for rocksdb. We currently
        // enforce all tables using nram to indicate their primary keys.
        elog(ERROR, "Primary key must be created for NRAM table \"%s\"",
             RelationGetRelationName(rel));
        return NIL;
    }

    foreach (lc, index_list) {
        Oid index_oid = lfirst_oid(lc);
        HeapTuple index_tuple =
            SearchSysCache1(INDEXRELID, ObjectIdGetDatum(index_oid));
        Form_pg_index index_form;
        if (!HeapTupleIsValid(index_tuple))
            elog(ERROR, "cache lookup failed for index %u", index_oid);

        index_form = (Form_pg_index)GETSTRUCT(index_tuple);

        if (index_form->indisprimary) {
            for (int i = 0; i < index_form->indnkeyatts; i++)
                key_attrs =
                    lappend_int(key_attrs, index_form->indkey.values[i]);
            ReleaseSysCache(index_tuple);
            break;
        }

        ReleaseSysCache(index_tuple);
    }

    list_free(index_list);

    if (key_attrs->length == 0)
        elog(ERROR, "no primary key found for relation \"%s\"",
             RelationGetRelationName(rel));
    return key_attrs;
}

static NRAMState *get_nram_state(Relation rel) {
    NRAMState *state = (NRAMState *)rel->rd_amcache;
    MemoryContext oldctx;
    List *indexatts;
    int i = 0;
    ListCell *lc;

    if (state && IS_VALID_NRAM_STATE(state)) return state;

    NRAM_TEST_INFO("refreshing the nram state");

    oldctx = MemoryContextSwitchTo(CacheMemoryContext);
    state = palloc(sizeof(NRAMState));
    state->magic = NRAM_STATE_MAGIC;

    if (!current_session_engine) {
        MemoryContextSwitchTo(TopMemoryContext);
        current_session_engine = (KVEngine *)rocksengine_open();
        MemoryContextSwitchTo(CacheMemoryContext);
    }

    state->engine = current_session_engine;

    indexatts = nram_get_primary_key_attrs(rel);
    state->nkeys = list_length(indexatts);
    state->key_attrs = palloc(sizeof(int) * state->nkeys);
    foreach (lc, indexatts) state->key_attrs[i++] = lfirst_int(lc);

    rel->rd_amcache = (void *)state;
    MemoryContextSwitchTo(oldctx);
    return state;
}

/* ------------------------------------------------------------------------
 * Slot related callbacks
 * ------------------------------------------------------------------------
 */

static const TupleTableSlotOps *nram_slot_callbacks(Relation relation) {
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;
    get_nram_state(relation);
    return &TTSOpsHeapTuple;  // only use nram for heap tuples.
}

/* ------------------------------------------------------------------------
 * Table scan related callbacks
 * ------------------------------------------------------------------------
 */

static TableScanDesc nram_beginscan(Relation relation, Snapshot snapshot,
                                    int nkeys, struct ScanKeyData *key,
                                    ParallelTableScanDesc parallel_scan,
                                    uint32 flags) {
    KVScanDesc scan = (KVScanDesc)palloc0(sizeof(KVScanDescData));
    NRAMState *state = get_nram_state(relation);
    // TODO: consider table inside min key setting.
    MemoryContext oldctx = MemoryContextSwitchTo(TopTransactionContext);
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;
    RelationIncrementReferenceCount(relation);
    scan->min_key = rocksengine_get_min_key(state->engine, relation->rd_id);
    scan->max_key = rocksengine_get_max_key(state->engine, relation->rd_id);

    scan->rs_base.rs_rd = relation;
    scan->engine_iterator = rocksengine_create_iterator(state->engine, true);
    scan->rs_base.rs_snapshot = snapshot;
    scan->rs_base.rs_nkeys = nkeys;
    scan->rs_base.rs_key = key;

    if (scan->min_key != NULL) {
        // In case the table is not empty, seek the iterator starting point.
        rocksengine_iterator_seek(scan->engine_iterator, scan->min_key);
    }

    MemoryContextSwitchTo(oldctx);
    return (TableScanDesc)scan;
}

static void nram_rescan(TableScanDesc sscan, struct ScanKeyData *key,
                        bool set_params, bool allow_strat, bool allow_sync,
                        bool allow_pagemode) {
    NRAM_INFO();
}

static void nram_endscan(TableScanDesc sscan) {
    KVScanDesc scan = (KVScanDesc)sscan;
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;
    RelationDecrementReferenceCount(sscan->rs_rd);
    
    if (scan->min_key)
        pfree(scan->min_key);
    if (scan->max_key)
        pfree(scan->max_key);
    rocksengine_iterator_destroy(scan->engine_iterator);
    pfree(scan);
}

static bool nram_getnextslot(TableScanDesc scan, ScanDirection direction,
                             TupleTableSlot *slot) {
    NRAMKey tkey;
    NRAMValue tvalue;
    HeapTuple tuple;
    KVScanDesc sscan;
    KVEngineIterator *it;
    MemoryContext oldctx = MemoryContextSwitchTo(TopTransactionContext);

    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;
    ExecClearTuple(slot);
    sscan = (KVScanDesc)scan;
    it = sscan->engine_iterator;
    if (direction != ForwardScanDirection)
        elog(WARNING,
             "[NRAM]: currently we only support forward scan direction. Got %d",
             direction);

    if (!it->is_valid(it)) {
        MemoryContextSwitchTo(oldctx);
        return false;
    }

    it->get(it, &tkey, &tvalue);
    if (tkey->tableOid != scan->rs_rd->rd_id) {
        // The end of table. Currently, we only support forward scan.
        Assert(tkey->tableOid > scan->rs_rd->rd_id);
        MemoryContextSwitchTo(oldctx);
        return false;
    }
    tuple =
        deserialize_nram_value_to_tuple(tvalue, sscan->rs_base.rs_rd->rd_att);
    ExecStoreHeapTuple(tuple, slot, false);

    it->next(it);
    MemoryContextSwitchTo(oldctx);
    return true;
}

/* ------------------------------------------------------------------------
 * Index scan related callbacks
 * ------------------------------------------------------------------------
 */

static IndexFetchTableData *nram_index_fetch_begin(Relation rel) {
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;
    return NULL;
}

static void nram_index_fetch_reset(IndexFetchTableData *scan) { NRAM_INFO(); }

static void nram_index_fetch_end(IndexFetchTableData *scan) { NRAM_INFO(); }

static bool nram_index_fetch_tuple(IndexFetchTableData *scan, ItemPointer tid,
                                   Snapshot snapshot, TupleTableSlot *slot,
                                   bool *call_again, bool *all_dead) {
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;
    return false;
}

/* ------------------------------------------------------------------------
 * Callbacks for manipulations of physical tuples.
 * ------------------------------------------------------------------------
 */

static void nram_insert(Relation relation, HeapTuple tup, CommandId cid,
                        int options, BulkInsertState bistate) {
    NRAMState *nram_state;
    TupleDesc tupdesc;
    NRAMKey tkey;
    NRAMValue tvalue;
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;

    nram_state = get_nram_state(relation);
    tupdesc = RelationGetDescr(relation);

    // Serialize key and value from tuple
    tkey = nram_key_serialize_from_tuple(tup, tupdesc, nram_state->key_attrs,
                                         nram_state->nkeys);
    tvalue = nram_value_serialize_from_tuple(tup, tupdesc);

    rocksengine_put(nram_state->engine, tkey, tvalue);

    // Cleanup memory if necessary
    pfree(tkey);
    pfree(tvalue);

    ASSERT_VALID_NRAM_STATE(nram_state);
}

static void nram_tuple_insert(Relation relation, TupleTableSlot *slot,
                              CommandId cid, int options,
                              BulkInsertState bistate) {
    bool shouldFree = true;
    HeapTuple tuple = ExecFetchSlotHeapTuple(slot, true, &shouldFree);
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;

    /* Update the tuple with table oid */
    slot->tts_tableOid = RelationGetRelid(relation);
    tuple->t_tableOid = slot->tts_tableOid;

    /* Perform the insertion, and copy the resulting ItemPointer */
    nram_insert(relation, tuple, cid, options, bistate);
    ItemPointerSet(&tuple->t_self, 0,
                   1);  // block = 0, offset = 1 for invalid block/offsets.
    ItemPointerCopy(&tuple->t_self, &slot->tts_tid);

    if (shouldFree) {
        heap_freetuple(tuple);
    }
}

static void nram_tuple_insert_speculative(Relation relation,
                                          TupleTableSlot *slot, CommandId cid,
                                          int options, BulkInsertState bistate,
                                          uint32 specToken) {
    NRAM_INFO();
}

static void nram_tuple_complete_speculative(Relation relation,
                                            TupleTableSlot *slot,
                                            uint32 specToken, bool succeeded) {
    NRAM_INFO();
}

static void nram_multi_insert(Relation relation, TupleTableSlot **slots,
                              int ntuples, CommandId cid, int options,
                              BulkInsertState bistate) {
    NRAM_INFO();
}

static TM_Result nram_tuple_delete(Relation relation, ItemPointer tid,
                                   CommandId cid, Snapshot snapshot,
                                   Snapshot crosscheck, bool wait,
                                   TM_FailureData *tmfd, bool changingPart) {
    NRAM_INFO();
    return TM_Ok;
}

static TM_Result nram_tuple_update(Relation relation, ItemPointer otid,
                                   TupleTableSlot *slot, CommandId cid,
                                   Snapshot snapshot, Snapshot crosscheck,
                                   bool wait, TM_FailureData *tmfd,
                                   LockTupleMode *lockmode,
                                   TU_UpdateIndexes *update_indexes) {
    NRAM_INFO();
    return TM_Ok;
}

static TM_Result nram_tuple_lock(Relation relation, ItemPointer tid,
                                 Snapshot snapshot, TupleTableSlot *slot,
                                 CommandId cid, LockTupleMode mode,
                                 LockWaitPolicy wait_policy, uint8 flags,
                                 TM_FailureData *tmfd) {
    NRAM_INFO();
    return TM_Ok;
}

static void nram_finish_bulk_insert(Relation relation, int options) {
    NRAM_INFO();
}

/* ------------------------------------------------------------------------
 * Callbacks for non-modifying operations on individual tuples.
 * ------------------------------------------------------------------------
 */

static bool nram_fetch_row_version(Relation relation, ItemPointer tid,
                                   Snapshot snapshot, TupleTableSlot *slot) {
    NRAM_INFO();
    return false;
}

static void nram_get_latest_tid(TableScanDesc sscan, ItemPointer tid) {
    NRAM_INFO();
}

static bool nram_tuple_tid_valid(TableScanDesc scan, ItemPointer tid) {
    NRAM_INFO();
    return false;
}

static bool nram_tuple_satisfies_snapshot(Relation rel, TupleTableSlot *slot,
                                          Snapshot snapshot) {
    NRAM_INFO();
    return false;
}

static TransactionId nram_index_delete_tuples(Relation rel,
                                              TM_IndexDeleteOp *delstate) {
    NRAM_INFO();
    return InvalidTransactionId;
}

/* ------------------------------------------------------------------------
 * DDL related callbacks.
 * ------------------------------------------------------------------------
 */

static void nram_relation_set_new_filelocator(Relation rel,
                                              const RelFileLocator *newrlocator,
                                              char persistence,
                                              TransactionId *freezeXid,
                                              MultiXactId *minmulti) {
    NRAM_INFO();
}

static void nram_relation_nontransactional_truncate(Relation rel) {
    NRAM_INFO();
}

static void nram_relation_copy_data(Relation rel,
                                    const RelFileLocator *newrlocator) {
    NRAM_INFO();
}

static void nram_relation_copy_for_cluster(
    Relation OldHeap, Relation NewHeap, Relation OldIndex, bool use_sort,
    TransactionId OldestXmin, TransactionId *xid_cutoff,
    MultiXactId *multi_cutoff, double *num_tuples, double *tups_vacuumed,
    double *tups_recently_dead) {
    NRAM_INFO();
}

static void nram_vacuum_rel(Relation rel, VacuumParams *params,
                            BufferAccessStrategy bstrategy) {
    NRAM_INFO();
}

static bool nram_scan_analyze_next_block(TableScanDesc scan,
                                         BlockNumber blockno,
                                         BufferAccessStrategy bstrategy) {
    NRAM_INFO();
    return false;
}

static bool nram_scan_analyze_next_tuple(TableScanDesc scan,
                                         TransactionId OldestXmin,
                                         double *liverows, double *deadrows,
                                         TupleTableSlot *slot) {
    NRAM_INFO();
    return false;
}

static double nram_index_build_range_scan(
    Relation heapRelation, Relation indexRelation, IndexInfo *indexInfo,
    bool allow_sync, bool anyvisible, bool progress, BlockNumber start_blockno,
    BlockNumber numblocks, IndexBuildCallback callback, void *callback_state,
    TableScanDesc scan) {
    NRAM_INFO();
    return 0;
}

static void nram_index_validate_scan(Relation heapRelation,
                                     Relation indexRelation,
                                     IndexInfo *indexInfo, Snapshot snapshot,
                                     ValidateIndexState *state) {
    NRAM_INFO();
}

/* ------------------------------------------------------------------------
 * Miscellaneous callbacks
 * ------------------------------------------------------------------------
 */

static bool nram_relation_needs_toast_table(Relation rel) {
    NRAM_INFO();
    return false;
}

static Oid nram_relation_toast_am(Relation rel) {
    NRAM_INFO();
    return InvalidOid;
}

static void nram_fetch_toast_slice(Relation toastrel, Oid valueid,
                                   int32 attrsize, int32 sliceoffset,
                                   int32 slicelength, struct varlena *result) {
    NRAM_INFO();
}

/* ------------------------------------------------------------------------
 * Planner callbacks
 * ------------------------------------------------------------------------
 */

static void nram_estimate_rel_size(Relation rel, int32 *attr_widths,
                                   BlockNumber *pages, double *tuples,
                                   double *allvisfrac) {
    NRAM_INFO();
    /* no data available */
    if (attr_widths) *attr_widths = 0;
    if (pages) *pages = 0;
    if (tuples) *tuples = 0;
    if (allvisfrac) *allvisfrac = 0;
}

/* ------------------------------------------------------------------------
 * Exector callbacks
 * ------------------------------------------------------------------------
 */

static bool nram_scan_bitmap_next_block(TableScanDesc scan,
                                        TBMIterateResult *tbmres) {
    NRAM_INFO();
    return false;
}

static bool nram_scan_bitmap_next_tuple(TableScanDesc scan,
                                        TBMIterateResult *tbmres,
                                        TupleTableSlot *slot) {
    NRAM_INFO();
    return false;
}

static bool nram_scan_sample_next_block(TableScanDesc scan,
                                        SampleScanState *scanstate) {
    NRAM_INFO();
    return false;
}

static bool nram_scan_sample_next_tuple(TableScanDesc scan,
                                        SampleScanState *scanstate,
                                        TupleTableSlot *slot) {
    NRAM_INFO();
    return false;
}

Datum nram_tableam_handler(PG_FUNCTION_ARGS);
PG_FUNCTION_INFO_V1(nram_tableam_handler);

static const TableAmRoutine nram_methods = {
    .type = T_TableAmRoutine,
    .slot_callbacks = nram_slot_callbacks,
    .scan_begin = nram_beginscan,
    .scan_end = nram_endscan,
    .scan_rescan = nram_rescan,
    .scan_getnextslot = nram_getnextslot,
    .parallelscan_estimate = table_block_parallelscan_estimate,
    .parallelscan_initialize = table_block_parallelscan_initialize,
    .parallelscan_reinitialize = table_block_parallelscan_reinitialize,
    .index_fetch_begin = nram_index_fetch_begin,
    .index_fetch_reset = nram_index_fetch_reset,
    .index_fetch_end = nram_index_fetch_end,
    .index_fetch_tuple = nram_index_fetch_tuple,
    .finish_bulk_insert = nram_finish_bulk_insert,
    .tuple_insert = nram_tuple_insert,
    .tuple_insert_speculative = nram_tuple_insert_speculative,
    .tuple_complete_speculative = nram_tuple_complete_speculative,
    .multi_insert = nram_multi_insert,
    .tuple_delete = nram_tuple_delete,
    .tuple_update = nram_tuple_update,
    .tuple_lock = nram_tuple_lock,
    .tuple_fetch_row_version = nram_fetch_row_version,
    .tuple_get_latest_tid = nram_get_latest_tid,
    .tuple_tid_valid = nram_tuple_tid_valid,
    .tuple_satisfies_snapshot = nram_tuple_satisfies_snapshot,
    .index_delete_tuples = nram_index_delete_tuples,
    .relation_set_new_filelocator = nram_relation_set_new_filelocator,
    .relation_nontransactional_truncate =
        nram_relation_nontransactional_truncate,
    .relation_copy_data = nram_relation_copy_data,
    .relation_copy_for_cluster = nram_relation_copy_for_cluster,
    .relation_vacuum = nram_vacuum_rel,
    .scan_analyze_next_block = nram_scan_analyze_next_block,
    .scan_analyze_next_tuple = nram_scan_analyze_next_tuple,
    .index_build_range_scan = nram_index_build_range_scan,
    .index_validate_scan = nram_index_validate_scan,
    .relation_size = table_block_relation_size,
    .relation_needs_toast_table = nram_relation_needs_toast_table,
    .relation_toast_am = nram_relation_toast_am,
    .relation_fetch_toast_slice = nram_fetch_toast_slice,
    .relation_estimate_size = nram_estimate_rel_size,
    .scan_sample_next_block = nram_scan_sample_next_block,
    .scan_sample_next_tuple = nram_scan_sample_next_tuple,
    .scan_bitmap_next_block = nram_scan_bitmap_next_block,
    .scan_bitmap_next_tuple = nram_scan_bitmap_next_tuple};

Datum nram_tableam_handler(PG_FUNCTION_ARGS) {
    PG_RETURN_POINTER(&nram_methods);
}

/* ------------------------------------------------------------------------
 * Unit tests
 * ------------------------------------------------------------------------
 */

PG_FUNCTION_INFO_V1(run_nram_tests);

Datum run_nram_tests(PG_FUNCTION_ARGS) {
    run_kv_serialization_test();
    PG_RETURN_VOID();
}

void _PG_init(void) { nram_register_xact_hook(); }

void _PG_fini(void) {
    nram_shutdown_session();
    nram_unregister_xact_hook();
}
