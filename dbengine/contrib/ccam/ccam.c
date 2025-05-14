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

#define CCAM_INFO() elog(INFO, "[CCAM] calling function %s", __func__)

PG_MODULE_MAGIC;

// ccam_make_virtual_slot helper function that creates an in-memory tuple.
static TupleTableSlot *ccam_make_virtual_slot(TupleDesc tupdesc, Datum *values,
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

static const TupleTableSlotOps *ccam_slot_callbacks(Relation relation) {
    CCAM_INFO();
    return &TTSOpsHeapTuple;  // only use ccam for heap tuples.
}

/* ------------------------------------------------------------------------
 * Table scan related callbacks
 * ------------------------------------------------------------------------
 */

static TableScanDesc ccam_beginscan(Relation relation, Snapshot snapshot,
                                    int nkeys, struct ScanKeyData *key,
                                    ParallelTableScanDesc parallel_scan,
                                    uint32 flags) {
    TableScanDesc sscan = (TableScanDesc)palloc0(sizeof(TableScanDescData));

    CCAM_INFO();
    sscan->rs_rd = relation;
    sscan->rs_snapshot = snapshot;
    sscan->rs_nkeys = nkeys;
    sscan->rs_key = key;
    sscan->rs_parallel = parallel_scan;
    sscan->rs_flags = flags;
    return (TableScanDesc)sscan;
}

static void ccam_rescan(TableScanDesc sscan, struct ScanKeyData *key,
                        bool set_params, bool allow_strat, bool allow_sync,
                        bool allow_pagemode) {
    CCAM_INFO();
}

static void ccam_endscan(TableScanDesc sscan) {
    CCAM_INFO();
    pfree(sscan);
}

static bool ccam_getnextslot(TableScanDesc sscan, ScanDirection direction,
                             TupleTableSlot *slot) {
    static bool returned = false;
    TupleDesc desc;
    TupleTableSlot *filled;
    Datum values[2];
    bool isnull[2];

    CCAM_INFO();
    ExecClearTuple(slot);
    if (returned) return false;

    returned = true;
    desc = slot->tts_tupleDescriptor;

    // TODO phx: implement the KV scan logic here.
    values[0] = Int32GetDatum(1);
    isnull[0] = false;
    values[1] = CStringGetTextDatum("hello ccam");
    isnull[1] = false;

    filled = ccam_make_virtual_slot(desc, values, isnull);
    ExecCopySlot(slot, filled);
    ExecDropSingleTupleTableSlot(filled);
    return true;
}

/* ------------------------------------------------------------------------
 * Index scan related callbacks
 * ------------------------------------------------------------------------
 */

static IndexFetchTableData *ccam_index_fetch_begin(Relation rel) {
    CCAM_INFO();
    return NULL;
}

static void ccam_index_fetch_reset(IndexFetchTableData *scan) { CCAM_INFO(); }

static void ccam_index_fetch_end(IndexFetchTableData *scan) { CCAM_INFO(); }

static bool ccam_index_fetch_tuple(IndexFetchTableData *scan, ItemPointer tid,
                                   Snapshot snapshot, TupleTableSlot *slot,
                                   bool *call_again, bool *all_dead) {
    CCAM_INFO();
    return false;
}

/* ------------------------------------------------------------------------
 * Callbacks for manipulations of physical tuples.
 * ------------------------------------------------------------------------
 */

static void ccam_tuple_insert(Relation relation, TupleTableSlot *slot,
                              CommandId cid, int options,
                              BulkInsertState bistate) {
    CCAM_INFO();
}

static void ccam_tuple_insert_speculative(Relation relation,
                                          TupleTableSlot *slot, CommandId cid,
                                          int options, BulkInsertState bistate,
                                          uint32 specToken) {
    CCAM_INFO();
}

static void ccam_tuple_complete_speculative(Relation relation,
                                            TupleTableSlot *slot,
                                            uint32 specToken, bool succeeded) {
    CCAM_INFO();
}

static void ccam_multi_insert(Relation relation, TupleTableSlot **slots,
                              int ntuples, CommandId cid, int options,
                              BulkInsertState bistate) {
    CCAM_INFO();
}

static TM_Result ccam_tuple_delete(Relation relation, ItemPointer tid,
                                   CommandId cid, Snapshot snapshot,
                                   Snapshot crosscheck, bool wait,
                                   TM_FailureData *tmfd, bool changingPart) {
    CCAM_INFO();
    return TM_Ok;
}

static TM_Result ccam_tuple_update(Relation relation, ItemPointer otid,
                                   TupleTableSlot *slot, CommandId cid,
                                   Snapshot snapshot, Snapshot crosscheck,
                                   bool wait, TM_FailureData *tmfd,
                                   LockTupleMode *lockmode,
                                   TU_UpdateIndexes *update_indexes) {
    CCAM_INFO();
    return TM_Ok;
}

static TM_Result ccam_tuple_lock(Relation relation, ItemPointer tid,
                                 Snapshot snapshot, TupleTableSlot *slot,
                                 CommandId cid, LockTupleMode mode,
                                 LockWaitPolicy wait_policy, uint8 flags,
                                 TM_FailureData *tmfd) {
    CCAM_INFO();
    return TM_Ok;
}

static void ccam_finish_bulk_insert(Relation relation, int options) {
    CCAM_INFO();
}

/* ------------------------------------------------------------------------
 * Callbacks for non-modifying operations on individual tuples.
 * ------------------------------------------------------------------------
 */

static bool ccam_fetch_row_version(Relation relation, ItemPointer tid,
                                   Snapshot snapshot, TupleTableSlot *slot) {
    CCAM_INFO();
    return false;
}

static void ccam_get_latest_tid(TableScanDesc sscan, ItemPointer tid) {
    CCAM_INFO();
}

static bool ccam_tuple_tid_valid(TableScanDesc scan, ItemPointer tid) {
    CCAM_INFO();
    return false;
}

static bool ccam_tuple_satisfies_snapshot(Relation rel, TupleTableSlot *slot,
                                          Snapshot snapshot) {
    CCAM_INFO();
    return false;
}

static TransactionId ccam_index_delete_tuples(Relation rel,
                                              TM_IndexDeleteOp *delstate) {
    CCAM_INFO();
    return InvalidTransactionId;
}

/* ------------------------------------------------------------------------
 * DDL related callbacks.
 * ------------------------------------------------------------------------
 */

static void ccam_relation_set_new_filelocator(Relation rel,
                                              const RelFileLocator *newrlocator,
                                              char persistence,
                                              TransactionId *freezeXid,
                                              MultiXactId *minmulti) {
    CCAM_INFO();
}

static void ccam_relation_nontransactional_truncate(Relation rel) {
    CCAM_INFO();
}

static void ccam_relation_copy_data(Relation rel,
                                    const RelFileLocator *newrlocator) {
    CCAM_INFO();
}

static void ccam_relation_copy_for_cluster(
    Relation OldHeap, Relation NewHeap, Relation OldIndex, bool use_sort,
    TransactionId OldestXmin, TransactionId *xid_cutoff,
    MultiXactId *multi_cutoff, double *num_tuples, double *tups_vacuumed,
    double *tups_recently_dead) {
    CCAM_INFO();
}

static void ccam_vacuum_rel(Relation rel, VacuumParams *params,
                            BufferAccessStrategy bstrategy) {
    CCAM_INFO();
}

static bool ccam_scan_analyze_next_block(TableScanDesc scan,
                                         BlockNumber blockno,
                                         BufferAccessStrategy bstrategy) {
    CCAM_INFO();
    return false;
}

static bool ccam_scan_analyze_next_tuple(TableScanDesc scan,
                                         TransactionId OldestXmin,
                                         double *liverows, double *deadrows,
                                         TupleTableSlot *slot) {
    CCAM_INFO();
    return false;
}

static double ccam_index_build_range_scan(
    Relation heapRelation, Relation indexRelation, IndexInfo *indexInfo,
    bool allow_sync, bool anyvisible, bool progress, BlockNumber start_blockno,
    BlockNumber numblocks, IndexBuildCallback callback, void *callback_state,
    TableScanDesc scan) {
    CCAM_INFO();
    return 0;
}

static void ccam_index_validate_scan(Relation heapRelation,
                                     Relation indexRelation,
                                     IndexInfo *indexInfo, Snapshot snapshot,
                                     ValidateIndexState *state) {
    CCAM_INFO();
}

/* ------------------------------------------------------------------------
 * Miscellaneous callbacks
 * ------------------------------------------------------------------------
 */

static bool ccam_relation_needs_toast_table(Relation rel) {
    CCAM_INFO();
    return false;
}

static Oid ccam_relation_toast_am(Relation rel) {
    CCAM_INFO();
    return InvalidOid;
}

static void ccam_fetch_toast_slice(Relation toastrel, Oid valueid,
                                   int32 attrsize, int32 sliceoffset,
                                   int32 slicelength, struct varlena *result) {
    CCAM_INFO();
}

/* ------------------------------------------------------------------------
 * Planner callbacks
 * ------------------------------------------------------------------------
 */

static void ccam_estimate_rel_size(Relation rel, int32 *attr_widths,
                                   BlockNumber *pages, double *tuples,
                                   double *allvisfrac) {
    CCAM_INFO();
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

static bool ccam_scan_bitmap_next_block(TableScanDesc scan,
                                        TBMIterateResult *tbmres) {
    CCAM_INFO();
    return false;
}

static bool ccam_scan_bitmap_next_tuple(TableScanDesc scan,
                                        TBMIterateResult *tbmres,
                                        TupleTableSlot *slot) {
    CCAM_INFO();
    return false;
}

static bool ccam_scan_sample_next_block(TableScanDesc scan,
                                        SampleScanState *scanstate) {
    CCAM_INFO();
    return false;
}

static bool ccam_scan_sample_next_tuple(TableScanDesc scan,
                                        SampleScanState *scanstate,
                                        TupleTableSlot *slot) {
    CCAM_INFO();
    return false;
}

Datum ccam_tableam_handler(PG_FUNCTION_ARGS);
PG_FUNCTION_INFO_V1(ccam_tableam_handler);

static const TableAmRoutine ccam_methods = {
    .type = T_TableAmRoutine,
    .slot_callbacks = ccam_slot_callbacks,
    .scan_begin = ccam_beginscan,
    .scan_end = ccam_endscan,
    .scan_rescan = ccam_rescan,
    .scan_getnextslot = ccam_getnextslot,
    .parallelscan_estimate = table_block_parallelscan_estimate,
    .parallelscan_initialize = table_block_parallelscan_initialize,
    .parallelscan_reinitialize = table_block_parallelscan_reinitialize,
    .index_fetch_begin = ccam_index_fetch_begin,
    .index_fetch_reset = ccam_index_fetch_reset,
    .index_fetch_end = ccam_index_fetch_end,
    .index_fetch_tuple = ccam_index_fetch_tuple,
    .finish_bulk_insert = ccam_finish_bulk_insert,
    .tuple_insert = ccam_tuple_insert,
    .tuple_insert_speculative = ccam_tuple_insert_speculative,
    .tuple_complete_speculative = ccam_tuple_complete_speculative,
    .multi_insert = ccam_multi_insert,
    .tuple_delete = ccam_tuple_delete,
    .tuple_update = ccam_tuple_update,
    .tuple_lock = ccam_tuple_lock,
    .tuple_fetch_row_version = ccam_fetch_row_version,
    .tuple_get_latest_tid = ccam_get_latest_tid,
    .tuple_tid_valid = ccam_tuple_tid_valid,
    .tuple_satisfies_snapshot = ccam_tuple_satisfies_snapshot,
    .index_delete_tuples = ccam_index_delete_tuples,
    .relation_set_new_filelocator = ccam_relation_set_new_filelocator,
    .relation_nontransactional_truncate =
        ccam_relation_nontransactional_truncate,
    .relation_copy_data = ccam_relation_copy_data,
    .relation_copy_for_cluster = ccam_relation_copy_for_cluster,
    .relation_vacuum = ccam_vacuum_rel,
    .scan_analyze_next_block = ccam_scan_analyze_next_block,
    .scan_analyze_next_tuple = ccam_scan_analyze_next_tuple,
    .index_build_range_scan = ccam_index_build_range_scan,
    .index_validate_scan = ccam_index_validate_scan,
    .relation_size = table_block_relation_size,
    .relation_needs_toast_table = ccam_relation_needs_toast_table,
    .relation_toast_am = ccam_relation_toast_am,
    .relation_fetch_toast_slice = ccam_fetch_toast_slice,
    .relation_estimate_size = ccam_estimate_rel_size,
    .scan_sample_next_block = ccam_scan_sample_next_block,
    .scan_sample_next_tuple = ccam_scan_sample_next_tuple,
    .scan_bitmap_next_block = ccam_scan_bitmap_next_block,
    .scan_bitmap_next_tuple = ccam_scan_bitmap_next_tuple};

Datum ccam_tableam_handler(PG_FUNCTION_ARGS) {
    PG_RETURN_POINTER(&ccam_methods);
}
