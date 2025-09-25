#include "nram.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "utils/memutils.h"
#include "postgres.h"
#include "fmgr.h"
#include "access/tableam.h"
#include "access/heapam.h"
#include "access/xact.h"
#include "access/htup_details.h"
#include "nodes/execnodes.h"
#include "catalog/index.h"
#include "commands/vacuum.h"
#include "utils/builtins.h"
#include "executor/tuptable.h"
#include "utils/elog.h"
#include "utils/snapmgr.h"
#include "storage/ipc.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "access/heapam.h"
#include "access/multixact.h"
#include "nram_storage/rocks_service.h"
#include "nram_storage/rocks_handler.h"

PG_MODULE_MAGIC;

/* ------------------------------------------------------------------------
 * Helper functions.
 * ------------------------------------------------------------------------
 */

void nram_shutdown_session(void) { CloseRespChannel(); }

static bool nram_value_check_visibility(NRAMValue tvalue, NRAMXactState xact) {
    if (tvalue->xact_id == xact->xact_id || !NRAMValueIsPrivate(tvalue)) {
        return true;
    }
    return false;
}

// Mark a tuple as always-visible to core MVCC (safe for OCC)
static inline void nram_mark_tuple_visible_always(HeapTuple tup) {
    HeapTupleHeader t = tup->t_data;

    HeapTupleHeaderSetXminFrozen(t);
    HeapTupleHeaderSetXmax(t, InvalidTransactionId);
    t->t_infomask |= HEAP_XMAX_INVALID;
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

static TableScanDesc nram_beginscan(Relation relation, Snapshot snapshot,
                                    int nkeys, struct ScanKeyData *key,
                                    ParallelTableScanDesc parallel_scan,
                                    uint32 flags) {
    KVScanDesc scan = (KVScanDesc)palloc0(sizeof(KVScanDescData));
    NRAMKey min_key, max_key;
    NRAMXactState xact = GetCurrentNRAMXact();
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;
    RelationIncrementReferenceCount(relation);

    /*
     * For now we ignore `nkeys` and `key` since we do full table scan only.
     * If supporting index-only scans or predicate pushdown later, use them.
     */
    scan->rs_base.rs_rd = relation;
    scan->rs_base.rs_snapshot = snapshot;
    scan->rs_base.rs_nkeys = nkeys;
    scan->rs_base.rs_key = key;
    scan->rs_base.rs_flags = flags;
    scan->rs_base.rs_parallel = parallel_scan;

    /*
     * Determine scan bounds (min_key, max_key).
     * For now, we scan the full range for this tableOid.
     */
    min_key = palloc0(sizeof(NRAMKeyData));
    max_key = palloc0(sizeof(NRAMKeyData));
    min_key->tableOid = relation->rd_id;
    min_key->tid = 0;
    max_key->tableOid = relation->rd_id + 1;
    max_key->tid = 0;
    scan->min_key = min_key;
    scan->max_key = max_key;

    if (nram_for_read(xact) && xact->action->detect_all) {
        nram_lock_for_scan(relation);
    }

    if (!RocksClientRangeScan(min_key, max_key, &scan->results_key,
                              &scan->results, &scan->result_count))
        elog(ERROR, "Get range failed");

    scan->cursor = 0;

    return (TableScanDesc)scan;
}

// TODO: check locks for scan.
static void nram_rescan(TableScanDesc sscan, struct ScanKeyData *key,
                        bool set_params, bool allow_strat, bool allow_sync,
                        bool allow_pagemode) {
    KVScanDesc scan = (KVScanDesc)palloc0(sizeof(KVScanDescData));
    NRAM_INFO();
    scan->cursor = 0;
}

static void nram_endscan(TableScanDesc sscan) {
    KVScanDesc scan = (KVScanDesc)sscan;
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;
    RelationDecrementReferenceCount(sscan->rs_rd);

    pfree(scan->min_key);
    pfree(scan->max_key);

    if (scan->result_count) {
        for (int i = 0; i < scan->result_count; i++) {
            pfree(scan->results_key[i]);
            pfree(scan->results[i]);
        }
        pfree(scan->results);
        pfree(scan->results_key);
    }

    pfree(scan);
}

static bool nram_getnextslot(TableScanDesc sscan, ScanDirection direction,
                             TupleTableSlot *slot) {
    KVScanDesc scan = (KVScanDesc)sscan;
    NRAMValue tvalue;
    NRAMKey tkey;
    HeapTuple tuple;
    NRAMXactState xact = GetCurrentNRAMXact();

    NRAM_INFO();

    ExecClearTuple(slot);

    if (direction != ForwardScanDirection) {
        elog(WARNING, "[NRAM] Only forward scan supported, got %d", direction);
        return false;
    }

    while (true) {
        if (scan->cursor >= scan->result_count) {
            return false;
        }

        Assert(scan->results_key != NULL);
        Assert(scan->results != NULL);
        Assert(scan->cursor < scan->result_count);

        tkey = scan->results_key[scan->cursor];
        tvalue = scan->results[scan->cursor];
        if (read_own_write(xact, tkey, &tvalue) ||
            read_own_read(xact, tkey, &tvalue)) {
            // The value is read/modified by the current transaction
            scan->cursor++;
            break;
        } else if (nram_value_check_visibility(tvalue, xact)) {
            // For detect all, we need to lock the tuple for read and reload the value.
            if (xact->action->detect_all) {
                ItemPointerData tid;
                nram_encode_tid(tkey->tid, &tid);

                nram_lock_for_read(scan->rs_base.rs_rd, &tid);
                tvalue = RocksClientGet(tkey);
            }

            // For read operations expose to users, we need to validate the later.
            if (nram_for_read(xact)) {
                add_read_set(xact, tkey, tvalue);
            }
            scan->cursor++;
            break;
        }
        // Otherwise, skip this version and continue to the next one.
        scan->cursor++;
    }

    tuple =
        deserialize_nram_value_to_tuple(tvalue, scan->rs_base.rs_rd->rd_att);
    nram_mark_tuple_visible_always(tuple);
    tuple->t_tableOid = scan->rs_base.rs_rd->rd_id;
    nram_encode_tid(tkey->tid, &tuple->t_self);
    ExecStoreHeapTuple(tuple, slot, true);

    Assert(slot != NULL);
    Assert(TTS_EMPTY(slot) == false);
    Assert(ItemPointerIsValid(&slot->tts_tid));
    Assert(slot->tts_tableOid == scan->rs_base.rs_rd->rd_id);
    NRAM_TEST_INFO("Stored tuple: tid block=%u offset=%u",
                   BlockIdGetBlockNumber(&(slot->tts_tid.ip_blkid)),
                   slot->tts_tid.ip_posid);

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
    return (IndexFetchTableData *)kvscan;
}

static void nram_index_fetch_reset(IndexFetchTableData *scan) { NRAM_INFO(); }

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
    *call_again = false;
    *all_dead = false;

    if (read_own_write(xact, tkey, &tvalue) ||
        read_own_read(xact, tkey, &tvalue)) {
        // The value is read/modified by the current transaction, just do
        // nothing.
    } else {
        if (xact->action->detect_all) {
            if (nram_for_modify(xact))
                nram_lock_for_write(scan->rel, tid);
            else if (nram_for_read(xact))
                nram_lock_for_read(scan->rel, tid);
        }

        tvalue = RocksClientGet(tkey);
        if (tvalue == NULL) {
            *all_dead = true;
            pfree(tkey);
            return false;
        }
        if (!nram_value_check_visibility(tvalue, xact)) {
            *all_dead = false;
            pfree(tkey);
            pfree(tvalue);
            return false;
        }
        if(nram_for_read(xact))
            add_read_set(xact, tkey, tvalue);
    }

    tuple =
        deserialize_nram_value_to_tuple(tvalue, kvscan->xs_base.rel->rd_att);
    nram_encode_tid(tkey->tid, &tuple->t_self);
    tuple->t_tableOid = kvscan->xs_base.rel->rd_id;
    nram_mark_tuple_visible_always(tuple);

    ExecClearTuple(slot);
    ExecStoreHeapTuple(tuple, slot, true);

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
    NRAMXactState xact = GetCurrentNRAMXact();
    // bool ok;
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;

    tupdesc = RelationGetDescr(relation);
    nram_generate_tid(tid);
    NRAM_TEST_INFO("nram tid generated as %lu", nram_decode_tid(tid));

    // Serialize key and value from tuple
    // TODO: resolve the table level conflicts (between the insert and the
    // scan).
    tkey = nram_key_from_tid(tup->t_tableOid, tid);
    tvalue = nram_value_serialize_from_tuple(tup, tupdesc);
    tvalue->flags |= NRAMF_PRIVATE;  // mark as private until commit

    if (xact->action->detect_all)
        nram_lock_for_write(relation, tid);

    if (!RocksClientPut(tkey, tvalue)) {
        elog(ERROR, "RocksClientPut failed in insert");
    }
    tvalue->flags &=
        ~NRAMF_PRIVATE;  // deferred release of private mark until commit
    add_write_set(xact, tkey, tvalue);
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

// We currently do not consider INSERT ON CONFLICT.
static void nram_tuple_insert_speculative(Relation relation,
                                          TupleTableSlot *slot, CommandId cid,
                                          int options, BulkInsertState bistate,
                                          uint32 specToken) {
    NRAM_UNSUPPORTED();
}

// We currently do not consider INSERT ON CONFLICT.
static void nram_tuple_complete_speculative(Relation relation,
                                            TupleTableSlot *slot,
                                            uint32 specToken, bool succeeded) {
    NRAM_UNSUPPORTED();
}

// TODO: batch the insert operation during network communication.
static void nram_multi_insert(Relation relation, TupleTableSlot **slots,
                              int ntuples, CommandId cid, int options,
                              BulkInsertState bistate) {
    TupleDesc tupdesc = RelationGetDescr(relation);
    Oid tableOid = RelationGetRelid(relation);
    NRAMKey tkey;
    NRAMValue tvalue;
    NRAMXactState xact = GetCurrentNRAMXact();

    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;

    for (int i = 0; i < ntuples; i++) {
        TupleTableSlot *slot = slots[i];
        HeapTuple tuple;
        ItemPointerData tid;
        bool shouldFree = true;

        slot_getallattrs(slot);

        // Set table OID
        slot->tts_tableOid = tableOid;

        tuple = ExecFetchSlotHeapTuple(slot, true, &shouldFree);
        tuple->t_tableOid = tableOid;

        nram_generate_tid(&tid);
        NRAM_TEST_INFO("nram tid generated as %lu (multi %d)",
                       nram_decode_tid(&tid), i);

        // Serialize and insert
        tkey = nram_key_from_tid(tableOid, &tid);
        tvalue = nram_value_serialize_from_tuple(tuple, tupdesc);
        tvalue->flags |= NRAMF_PRIVATE;  // mark as private until commit

        if (xact->action->detect_all) {
            nram_lock_for_write(relation, &tid);
        }
        if (!RocksClientPut(tkey, tvalue)) {
            // The index conflicts are handled with table locks.
            elog(ERROR, "RocksClientPut failed in multi_insert");
        }

        tvalue->flags &=
            ~NRAMF_PRIVATE;  // deferred release of private mark until commit
        NRAM_TEST_INFO("ADD! key = <%d:%lu>", tkey->tableOid, tkey->tid);
        add_write_set(xact, tkey, tvalue);

        // Update tuple/slot with tid
        tuple->t_self = tid;
        slot->tts_tid = tid;

        pfree(tkey);
        pfree(tvalue);
        if (shouldFree) heap_freetuple(tuple);
    }
}

static TM_Result nram_tuple_delete(Relation relation, ItemPointer tid,
                                   CommandId cid, Snapshot snapshot,
                                   Snapshot crosscheck, bool wait,
                                   TM_FailureData *tmfd, bool changingPart) {
    NRAM_UNSUPPORTED();
    return TM_Ok;
}

static TM_Result nram_tuple_update(Relation relation, ItemPointer otid,
                                   TupleTableSlot *slot, CommandId cid,
                                   Snapshot snapshot, Snapshot crosscheck,
                                   bool wait, TM_FailureData *tmfd,
                                   LockTupleMode *lockmode,
                                   TU_UpdateIndexes *update_indexes) {
    TupleDesc tupdesc = RelationGetDescr(relation);
    Oid tableOid = RelationGetRelid(relation);
    NRAMXactState xact = GetCurrentNRAMXact();
    NRAMKey new_key;
    NRAMValue new_value;
    HeapTuple new_tuple = NULL;
    ItemPointerData ntid =
        *otid;  // We maintain a single version, so otid == ntid
    bool shouldFree = true;

    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;

    // We do not support index now (to be supported)
    *lockmode = NoLock;
    *update_indexes = false;

    memset(tmfd, 0, sizeof(TM_FailureData));
    new_key = nram_key_from_tid(tableOid, &ntid);

    if (!ItemPointerIsValid(otid)) {
        elog(ERROR, "Invalid otid in nram_tuple_update");
    }

    {  // read the old value
        NRAMKey old_key = nram_key_from_tid(tableOid, otid);
        NRAMValue old_value = NULL;
        bool local_read = read_own_write(xact, old_key, &old_value);

        if (!local_read && xact->action->detect_all) {
            nram_lock_for_write(relation, otid);
        }
    }

    // For OCC, the update operation does not need to check visibility.
    // Just storage the new value into the local write set.
    slot_getallattrs(slot);
    slot->tts_tableOid = tableOid;
    slot->tts_tid = ntid;

    // calculate the new value
    new_tuple = ExecFetchSlotHeapTuple(slot, true, &shouldFree);
    new_tuple->t_tableOid = tableOid;
    new_tuple->t_self = ntid;
    new_value = nram_value_serialize_from_tuple(new_tuple, tupdesc);

    add_write_set(xact, new_key, new_value);

    if (shouldFree) heap_freetuple(new_tuple);
    pfree(new_key);
    pfree(new_value);
    NRAM_INFO();

    return TM_Ok;
}

static TM_Result nram_tuple_lock(Relation relation, ItemPointer tid,
                                 Snapshot snapshot, TupleTableSlot *slot,
                                 CommandId cid, LockTupleMode mode,
                                 LockWaitPolicy wait_policy, uint8 flags,
                                 TM_FailureData *tmfd) {
    NRAM_UNSUPPORTED();
    return TM_Ok;
}

static void nram_finish_bulk_insert(Relation relation, int options) {
    NRAM_UNSUPPORTED();
}

/* ------------------------------------------------------------------------
 * Callbacks for non-modifying operations on individual tuples.
 * ------------------------------------------------------------------------
 */

/*
Fetch tuple at `tid` into `slot`, after doing a visibility test according
to `snapshot`. If a tuple was found and passed the visibility test, then
returns true, false otherwise.

In NRAM, there should only exists one single version on kv storage (no visibity
prob). Thus, directly fetch the record and return true.
*/
static bool nram_fetch_row_version(Relation relation, ItemPointer tid,
                                   Snapshot snapshot, TupleTableSlot *slot) {
    NRAMKey tkey;
    NRAMValue tvalue;
    HeapTuple tuple;
    NRAMXactState xact = GetCurrentNRAMXact();
    NRAM_INFO();
    NRAM_XACT_BEGIN_BLOCK;

    tkey = nram_key_from_tid(relation->rd_id, tid);
    if (xact->action->detect_all) {
        if (nram_for_modify(xact))
            nram_lock_for_write(relation, tid);
        else if (nram_for_read(xact))
            nram_lock_for_read(relation, tid);
    }

    if (!read_own_write(xact, tkey, &tvalue) &&
        !read_own_read(xact, tkey, &tvalue)) {
        tvalue = RocksClientGet(tkey);
        if (tvalue == NULL) {
            pfree(tkey);
            return false;
        }
        if (!nram_value_check_visibility(tvalue, xact)) {
            pfree(tkey);
            pfree(tvalue);
            return false;
        }
        add_read_set(xact, tkey, tvalue);
        NRAM_TEST_INFO("readed data from transaction %u", tvalue->xact_id);
    }

    tuple = deserialize_nram_value_to_tuple(tvalue, relation->rd_att);
    nram_mark_tuple_visible_always(tuple);
    nram_encode_tid(tkey->tid, &tuple->t_self);
    tuple->t_tableOid = relation->rd_id;

    ExecClearTuple(slot);
    ExecStoreHeapTuple(tuple, slot, true);

    pfree(tkey);
    pfree(tvalue);
    return true;
}

static void nram_get_latest_tid(TableScanDesc sscan, ItemPointer tid) {
    NRAM_UNSUPPORTED();
}

static bool nram_tuple_tid_valid(TableScanDesc scan, ItemPointer tid) {
    NRAM_UNSUPPORTED();
    return false;
}

static bool nram_tuple_satisfies_snapshot(Relation rel, TupleTableSlot *slot,
                                          Snapshot snapshot) {
    // single-version OCC: anything we return is already filtered
    // (committed store || write-set overlay).
    return !TTS_EMPTY(slot);
}

static TransactionId nram_index_delete_tuples(Relation rel,
                                              TM_IndexDeleteOp *delstate) {
    NRAM_UNSUPPORTED();
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
    /*
     * Initialize to the minimum XID that could put tuples in the table. We
     * know that no xacts older than RecentXmin are still running, so that
     * will do.
     */
    *freezeXid = RecentXmin;

    /*
     * Similarly, initialize the minimum Multixact to the first value that
     * could possibly be stored in tuples in the table.  Running transactions
     * could reuse values from their local cache, so we are careful to
     * consider all currently running multis.
     *
     * XXX this could be refined further, but is it worth the hassle?
     */
    *minmulti = GetOldestMultiXactId();
}

static void nram_relation_nontransactional_truncate(Relation rel) {
    NRAM_UNSUPPORTED();
}

static void nram_relation_copy_data(Relation rel,
                                    const RelFileLocator *newrlocator) {
    NRAM_UNSUPPORTED();
}

static void nram_relation_copy_for_cluster(
    Relation OldHeap, Relation NewHeap, Relation OldIndex, bool use_sort,
    TransactionId OldestXmin, TransactionId *xid_cutoff,
    MultiXactId *multi_cutoff, double *num_tuples, double *tups_vacuumed,
    double *tups_recently_dead) {
    NRAM_UNSUPPORTED();
}

static void nram_vacuum_rel(Relation rel, VacuumParams *params,
                            BufferAccessStrategy bstrategy) {
    NRAM_UNSUPPORTED();
}

static bool nram_scan_analyze_next_block(TableScanDesc scan,
                                         BlockNumber blockno,
                                         BufferAccessStrategy bstrategy) {
    NRAM_UNSUPPORTED();
    return false;
}

static bool nram_scan_analyze_next_tuple(TableScanDesc scan,
                                         TransactionId OldestXmin,
                                         double *liverows, double *deadrows,
                                         TupleTableSlot *slot) {
    NRAM_UNSUPPORTED();
    return false;
}

static double nram_index_build_range_scan(
    Relation heapRelation, Relation indexRelation, IndexInfo *indexInfo,
    bool allow_sync, bool anyvisible, bool progress, BlockNumber start_blockno,
    BlockNumber numblocks, IndexBuildCallback callback, void *callback_state,
    TableScanDesc scan) {
    NRAM_INFO();
    // return heapam_index_build_range_scan(
    //     heapRelation, indexRelation, indexInfo, allow_sync, anyvisible,
    //     progress, start_blockno, numblocks, callback, callback_state, scan);
    return 0;
}

static void nram_index_validate_scan(Relation heapRelation,
                                     Relation indexRelation,
                                     IndexInfo *indexInfo, Snapshot snapshot,
                                     ValidateIndexState *state) {
    NRAM_UNSUPPORTED();
}

/* ------------------------------------------------------------------------
 * Miscellaneous callbacks (we haven't supported the relation toast yet)
 * ------------------------------------------------------------------------
 */

static bool nram_relation_needs_toast_table(Relation rel) { return false; }

static Oid nram_relation_toast_am(Relation rel) {
    NRAM_UNSUPPORTED();
    return InvalidOid;
}

static void nram_fetch_toast_slice(Relation toastrel, Oid valueid,
                                   int32 attrsize, int32 sliceoffset,
                                   int32 slicelength, struct varlena *result) {
    NRAM_UNSUPPORTED();
}

/* ------------------------------------------------------------------------
 * Planner callbacks
 * ------------------------------------------------------------------------
 */

static void nram_estimate_rel_size(Relation rel, int32 *attr_widths,
                                   BlockNumber *pages, double *tuples,
                                   double *allvisfrac) {
    // Naive implementation, to be refined during plan optimizer implementation.
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
    NRAM_UNSUPPORTED();
    return false;
}

static bool nram_scan_bitmap_next_tuple(TableScanDesc scan,
                                        TBMIterateResult *tbmres,
                                        TupleTableSlot *slot) {
    NRAM_UNSUPPORTED();
    return false;
}

static bool nram_scan_sample_next_block(TableScanDesc scan,
                                        SampleScanState *scanstate) {
    NRAM_UNSUPPORTED();
    return false;
}

static bool nram_scan_sample_next_tuple(TableScanDesc scan,
                                        SampleScanState *scanstate,
                                        TupleTableSlot *slot) {
    NRAM_UNSUPPORTED();
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
    run_kv_copy_test();
    run_kv_rocks_service_basic_test();
    run_kv_rocks_client_get_put_test();
    run_kv_rocks_client_range_scan_test();

    run_channel_basic_test();
    run_channel_sequential_test();
    run_channel_multiprocess_test();
    run_channel_msg_basic_test();

    run_policy_basic_test();
    run_policy_full_load_verify_test();
    PG_RETURN_VOID();
}

static ExecutorStart_hook_type prev_ExecutorStart = NULL;
static ExecutorEnd_hook_type prev_ExecutorEnd = NULL;

static void nram_ExecutorStart(QueryDesc *qd, int eflags) {
    NRAMXactState x = GetCurrentNRAMXact();
    CmdType ct = qd ? qd->operation : CMD_UNKNOWN;
    if (x->cmdtype_depth < NRAM_CMD_STACK_MAX)
        x->cmdtype_stack[x->cmdtype_depth] = ct;
    x->cmdtype_depth = Min(x->cmdtype_depth + 1, NRAM_CMD_STACK_MAX);
    x->cur_cmdtype = ct;
    before_access(x);

    if (prev_ExecutorStart)
        prev_ExecutorStart(qd, eflags);
    else
        standard_ExecutorStart(qd, eflags);
}

static void nram_ExecutorEnd(QueryDesc *qd) {
    NRAMXactState x = GetCurrentNRAMXact();
    if (x && x->cmdtype_depth > 0) {
        x->cmdtype_depth--;
        x->cur_cmdtype = (x->cmdtype_depth > 0)
                             ? x->cmdtype_stack[x->cmdtype_depth - 1]
                             : CMD_UNKNOWN;
    }

    if (prev_ExecutorEnd)
        prev_ExecutorEnd(qd);
    else
        standard_ExecutorEnd(qd);
}

shmem_request_hook_type prev_shmem_request_hook = NULL;
shmem_startup_hook_type prev_shmem_startup_hook = NULL;

static void nram_shmem_request(void) {
    NRAM_INFO();
    if (prev_shmem_request_hook) prev_shmem_request_hook();

    RequestAddinShmemSpace(sizeof(KVChannelShared) * (MAX_PROC_COUNT + 1));
    RequestAddinShmemSpace(sizeof(CachedAgentFuncData));
}

static void nram_shmem_startup(void) {
    if (prev_shmem_startup_hook) prev_shmem_startup_hook();

    /* Serialize init of shared structs */
    LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
    init_agent_function_cache(); /* <— do it here */
    /* init any other ShmemInitStruct-backed objects here as well */
    LWLockRelease(AddinShmemInitLock);
}

void _PG_init(void) {
    prev_shmem_request_hook = shmem_request_hook;
    shmem_request_hook = nram_shmem_request;
    prev_shmem_startup_hook = shmem_startup_hook;
    shmem_startup_hook = nram_shmem_startup;
    prev_ExecutorStart = ExecutorStart_hook;
    ExecutorStart_hook = nram_ExecutorStart;
    prev_ExecutorEnd   = ExecutorEnd_hook;
    ExecutorEnd_hook   = nram_ExecutorEnd;

    nram_init();
    nram_register_xact_hook();
    nram_rocks_service_init();
}

void _PG_fini(void) {
    nram_shutdown_session();
    nram_unregister_xact_hook();
    nram_rocks_service_terminate();
    shmem_request_hook = prev_shmem_request_hook;
    shmem_startup_hook = prev_shmem_startup_hook;
    ExecutorStart_hook = prev_ExecutorStart;
    ExecutorEnd_hook   = prev_ExecutorEnd;
}

/* ------------------------------------------------------------------------
 * Load agent function
 * ------------------------------------------------------------------------
 */

Datum nram_load_policy(PG_FUNCTION_ARGS);
PG_FUNCTION_INFO_V1(nram_load_policy);

Datum nram_load_policy(PG_FUNCTION_ARGS) {
    text *tpath = PG_GETARG_TEXT_PP(0);
    char *path = text_to_cstring(tpath);

    load_agent_function(path);
    PG_RETURN_VOID();
}
