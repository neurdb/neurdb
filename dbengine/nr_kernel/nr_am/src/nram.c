#include "postgres.h"
#include "fmgr.h"
#include "access/tableam.h"
#include "access/heapam.h"
#include "nodes/execnodes.h"
#include "catalog/index.h"
#include "commands/vacuum.h"
#include "utils/builtins.h"
#include "executor/tuptable.h"
#include "utils/elog.h"

#define NRAM_INFO() elog(INFO, "[NRAM] calling function %s", __func__)

PG_MODULE_MAGIC;

// nram_make_virtual_slot helper function that creates an in-memory tuple.
static TupleTableSlot *nram_make_virtual_slot(TupleDesc tupdesc, Datum *values,
                                              bool *isnull) {
    TupleTableSlot *slot;
    slot = MakeSingleTupleTableSlot(tupdesc, &TTSOpsVirtual);
    for (int i = 0; i < tupdesc->natts; i++) {
        slot->tts_values[i] = values[i];
        slot->tts_isnull[i] = isnull[i];
    }
    ExecStoreVirtualTuple(slot);
    return slot;
}

/* ------------------------------------------------------------------------
 * Slot related callbacks
 * ------------------------------------------------------------------------
 */

static const TupleTableSlotOps *nram_slot_callbacks(Relation relation) {
    NRAM_INFO();
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
    TableScanDesc sscan = (TableScanDesc)palloc0(sizeof(TableScanDescData));

    NRAM_INFO();
    sscan->rs_rd = relation;
    sscan->rs_snapshot = snapshot;
    sscan->rs_nkeys = nkeys;
    sscan->rs_key = key;
    sscan->rs_parallel = parallel_scan;
    sscan->rs_flags = flags;
    return (TableScanDesc)sscan;
}

static void nram_rescan(TableScanDesc sscan, struct ScanKeyData *key,
                        bool set_params, bool allow_strat, bool allow_sync,
                        bool allow_pagemode) {
    NRAM_INFO();
}

static void nram_endscan(TableScanDesc sscan) {
    NRAM_INFO();
    pfree(sscan);
}

static bool nram_getnextslot(TableScanDesc sscan, ScanDirection direction,
                             TupleTableSlot *slot) {
    static bool returned = false;
    TupleDesc desc;
    TupleTableSlot *filled;
    Datum values[2];
    bool isnull[2];

    NRAM_INFO();
    ExecClearTuple(slot);
    if (returned) return false;

    returned = true;
    desc = slot->tts_tupleDescriptor;

    // TODO phx: implement the KV scan logic here.
    values[0] = Int32GetDatum(1);
    isnull[0] = false;
    values[1] = CStringGetTextDatum("hello nram");
    isnull[1] = false;

    filled = nram_make_virtual_slot(desc, values, isnull);
    ExecCopySlot(slot, filled);
    ExecDropSingleTupleTableSlot(filled);
    return true;
}

/* ------------------------------------------------------------------------
 * Index scan related callbacks
 * ------------------------------------------------------------------------
 */

static IndexFetchTableData *nram_index_fetch_begin(Relation rel) {
    NRAM_INFO();
    return NULL;
}

static void nram_index_fetch_reset(IndexFetchTableData *scan) { NRAM_INFO(); }

static void nram_index_fetch_end(IndexFetchTableData *scan) { NRAM_INFO(); }

static bool nram_index_fetch_tuple(IndexFetchTableData *scan, ItemPointer tid,
                                   Snapshot snapshot, TupleTableSlot *slot,
                                   bool *call_again, bool *all_dead) {
    NRAM_INFO();
    return false;
}

/* ------------------------------------------------------------------------
 * Callbacks for manipulations of physical tuples.
 * ------------------------------------------------------------------------
 */

static void nram_tuple_insert(Relation relation, TupleTableSlot *slot,
                              CommandId cid, int options,
                              BulkInsertState bistate) {
    NRAM_INFO();
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
