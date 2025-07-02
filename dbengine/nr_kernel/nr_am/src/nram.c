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
#include "storage/ipc.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "nram_storage/rocks_service.h"

PG_MODULE_MAGIC;

/* ------------------------------------------------------------------------
 * Helper functions.
 * ------------------------------------------------------------------------
 */

void nram_shutdown_session(void) {
    KVEngine* engine = GetCurrentEngine();
    engine->destroy(engine);
}

/* ------------------------------------------------------------------------
 * Slot related callbacks
 * ------------------------------------------------------------------------
 */

static const TupleTableSlotOps *nram_slot_callbacks(Relation relation) {
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;
    return &TTSOpsHeapTuple;  // only use nram for heap tuples.
}

/* ------------------------------------------------------------------------
 * Table scan related callbacks
 * ------------------------------------------------------------------------
 */

// static uint64 operationId = 0;  /* a SQL might cause multiple scans */

static TableScanDesc nram_beginscan(Relation relation, Snapshot snapshot,
                                    int nkeys, struct ScanKeyData *key,
                                    ParallelTableScanDesc parallel_scan,
                                    uint32 flags) {
    KVScanDesc scan = (KVScanDesc)palloc0(sizeof(KVScanDescData));
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;

    // scan->read_state = palloc0(sizeof(TableReadState));
    // scan->read_state->execExplainOnly = flags & EXEC_FLAG_EXPLAIN_ONLY? true: false;
    // scan->read_state->operationId = 0;
    // scan->read_state->done = false;
    // scan->read_state->key = NULL;
    
    RelationIncrementReferenceCount(relation);
    scan->rs_base.rs_rd = relation;
    scan->rs_base.rs_snapshot = snapshot;
    scan->rs_base.rs_nkeys = nkeys;
    scan->rs_base.rs_key = key;


    // TODO: send the args to the remote server.
    // ReadBatchArgs args;
    // args.buf = &readState->buf;
    // args.bufLen = &readState->bufLen;
    // args.opid = ++operationId;
    // readState->hasNext = KVReadBatchRequest(relationId, &args);
    // rocksengine_get();    
    // scan->read_state->next = scan->read_state->buf;
    // scan->read_state->operationId = operationId;
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
    rocksengine_iterator_destroy(GetCurrentEngine(), scan->engine_iterator);
    pfree(scan);
}

// TODO: the get next here could cause conflict on the index!
// TODO: filter out those uncommitted records.
static bool nram_getnextslot(TableScanDesc scan, ScanDirection direction,
                             TupleTableSlot *slot) {
    NRAMKey tkey;
    NRAMValue tvalue;
    HeapTuple tuple;
    KVScanDesc sscan;
    KVEngineIterator *it;
    MemoryContext oldctx = MemoryContextSwitchTo(TopTransactionContext);
    NRAMXactState xact = GetCurrentNRAMXact();

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
    if (!read_own_write(xact, tkey, &tvalue) && !read_own_read(xact, tkey, &tvalue)) {
        add_read_set(xact, tkey, tvalue);
    }

    if (tkey->tableOid != scan->rs_rd->rd_id) {
        // The end of table. Currently, we only support forward scan.
        // Not that depending on the machine, big/small endian.
        // The tkey->tableOid could be smaller than scan->rs_rd->rd_id. This is expected.
        pfree(tkey);
        pfree(tvalue);
        MemoryContextSwitchTo(oldctx);
        return false;
    }
    tuple =
        deserialize_nram_value_to_tuple(tvalue, sscan->rs_base.rs_rd->rd_att);
    ExecStoreHeapTuple(tuple, slot, true);

    it->next(it);

    pfree(tkey);
    pfree(tvalue);
    // NRAM_INFO();
    // heap_freetuple(tuple);

    MemoryContextSwitchTo(oldctx);
    return true;
}

/* ------------------------------------------------------------------------
 * Index scan related callbacks
 * ------------------------------------------------------------------------
 */

static IndexFetchTableData *nram_index_fetch_begin(Relation relation) {
    IndexFetchKVData *kvscan;
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;
    kvscan = palloc0(sizeof(IndexFetchKVData));
    kvscan->xs_base.rel = relation;
    kvscan->xs_engine = GetCurrentEngine();
    return (IndexFetchTableData *)kvscan;
}

static void nram_index_fetch_reset(IndexFetchTableData *scan) {
    NRAM_INFO();
}

static void nram_index_fetch_end(IndexFetchTableData *scan) {
    IndexFetchKVData *kvscan;
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;
    kvscan = (IndexFetchKVData *)scan;
    nram_index_fetch_reset(scan);
    pfree(kvscan);
}

static bool nram_index_fetch_tuple(IndexFetchTableData *scan, ItemPointer tid,
                                   Snapshot snapshot, TupleTableSlot *slot,
                                   bool *call_again, bool *all_dead) {
    IndexFetchKVData *kvscan;
    NRAMKey tkey;
    NRAMValue tvalue;
    HeapTuple tuple;
    NRAMXactState xact = GetCurrentNRAMXact();
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;

    kvscan = (IndexFetchKVData *)scan;
    tkey = nram_key_from_tid(kvscan->xs_base.rel->rd_id, tid);
    if (!read_own_write(xact, tkey, &tvalue) && !read_own_read(xact, tkey, &tvalue)) {
        tvalue = rocksengine_get(kvscan->xs_engine, tkey);
        if (tvalue == NULL) {
            pfree(tkey);
            return false;
        }
        add_read_set(xact, tkey, tvalue);
    }

    tuple = deserialize_nram_value_to_tuple(
        tvalue,
        kvscan->xs_base.rel->rd_att
    );

    ExecClearTuple(slot);
    ExecStoreHeapTuple(tuple, slot, true);
    *call_again = false;
    *all_dead = false;

    pfree(tkey);
    pfree(tvalue);
    return true;
}

/* ------------------------------------------------------------------------
 * Callbacks for manipulations of physical tuples.
 * ------------------------------------------------------------------------
 */

static void nram_insert(Relation relation, HeapTuple tup, CommandId cid,
                        int options, BulkInsertState bistate, ItemPointer tid) {
    TupleDesc tupdesc;
    NRAMKey tkey;
    NRAMValue tvalue;
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;

    tupdesc = RelationGetDescr(relation);
    nram_generate_tid(tid);
    NRAM_TEST_INFO("nram tid generated as %lu", nram_decode_tid(tid));

    // Serialize key and value from tuple
    tkey = nram_key_from_tid(tup->t_tableOid, tid);
    tvalue = nram_value_serialize_from_tuple(tup, tupdesc);

    rocksengine_put(GetCurrentEngine(), tkey, tvalue);
    add_write_set(GetCurrentNRAMXact(), tkey, tvalue);
    NRAM_TEST_INFO("ADD! key = <%d:%lu>", tkey->tableOid, tkey->tid);

    // Cleanup memory if necessary.
    pfree(tkey);
    pfree(tvalue);

}

static void nram_tuple_insert(Relation relation, TupleTableSlot *slot,
                              CommandId cid, int options,
                              BulkInsertState bistate) {
    bool shouldFree = true;
    HeapTuple tuple = ExecFetchSlotHeapTuple(slot, true, &shouldFree);
    ItemPointerData tid;

    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;

    slot_getallattrs(slot);

    /* Update the tuple with table oid */
    slot->tts_tableOid = RelationGetRelid(relation);
    tuple->t_tableOid = slot->tts_tableOid;

    /* Perform the insertion, and copy the unique resulting ItemPointer tid */
    nram_insert(relation, tuple, cid, options, bistate, &tid);
    Assert(ItemPointerIsValid(&tid));
    Assert(tuple != NULL);
    Assert(slot != NULL);
    tuple->t_self = tid;
    slot->tts_tid = tid;

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
    // run_kv_serialization_test();
    // run_kv_copy_test();
    run_kv_rocks_service_basic_test();
    
    // run_channel_basic_test();
    // run_channel_sequential_test();
    // run_channel_multiprocess_test();
    // run_channel_msg_basic_test();
    PG_RETURN_VOID();
}

void _PG_init(void) {
    // prev_shmem_startup_hook = shmem_startup_hook;
    // shmem_startup_hook = nram_shmem_startup;
    nram_init();
    nram_register_xact_hook();
    nram_rocks_service_init();
}

void _PG_fini(void) {
    nram_shutdown_session();
    nram_unregister_xact_hook();
    nram_rocks_service_terminate();
    // shmem_startup_hook = prev_shmem_startup_hook;
}
