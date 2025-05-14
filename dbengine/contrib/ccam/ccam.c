#include "postgres.h"
#include "fmgr.h"
#include "access/tableam.h"
#include "access/heapam.h"
#include "nodes/execnodes.h"
#include "catalog/index.h"
#include "commands/vacuum.h"
#include "utils/builtins.h"
#include "executor/tuptable.h"
#include "utils/elog.h"  // Needed for elog(WARNING/ERROR/...)

PG_MODULE_MAGIC;

static const TupleTableSlotOps* ccam_slot_callbacks(Relation relation) {
  elog(DEBUG1, "ccam_slot_callbacks called");
  return NULL;
}

static TableScanDesc ccam_beginscan(Relation relation, Snapshot snapshot, int nkeys, struct ScanKeyData *key, ParallelTableScanDesc parallel_scan, uint32 flags) {
  elog(DEBUG1, "ccam_beginscan called");
  return NULL;
}

static void ccam_rescan(TableScanDesc sscan, struct ScanKeyData *key, bool set_params, bool allow_strat, bool allow_sync, bool allow_pagemode) {
  elog(DEBUG1, "ccam_rescan called");
}

static void ccam_endscan(TableScanDesc sscan) {
  elog(DEBUG1, "ccam_endscan called");
}

static bool ccam_getnextslot(TableScanDesc sscan, ScanDirection direction, TupleTableSlot *slot) {
  elog(DEBUG1, "ccam_getnextslot called");
  return false;
}

static IndexFetchTableData* ccam_index_fetch_begin(Relation rel) {
  elog(DEBUG1, "ccam_index_fetch_begin called");
  return NULL;
}

static void ccam_index_fetch_reset(IndexFetchTableData *scan) {
  elog(DEBUG1, "ccam_index_fetch_reset called");
}

static void ccam_index_fetch_end(IndexFetchTableData *scan) {
  elog(DEBUG1, "ccam_index_fetch_end called");
}

static bool ccam_index_fetch_tuple(IndexFetchTableData *scan, ItemPointer tid, Snapshot snapshot, TupleTableSlot *slot, bool *call_again, bool *all_dead) {
  elog(DEBUG1, "ccam_index_fetch_tuple called");
  return false;
}

static void ccam_tuple_insert(Relation relation, TupleTableSlot *slot, CommandId cid, int options, BulkInsertState bistate) {
  elog(DEBUG1, "ccam_tuple_insert called");
}

static void ccam_tuple_insert_speculative(Relation relation, TupleTableSlot *slot, CommandId cid, int options, BulkInsertState bistate, uint32 specToken) {
  elog(DEBUG1, "ccam_tuple_insert_speculative called");
}

static void ccam_tuple_complete_speculative(Relation relation, TupleTableSlot *slot, uint32 specToken, bool succeeded) {
  elog(DEBUG1, "ccam_tuple_complete_speculative called");
}

static void ccam_multi_insert(Relation relation, TupleTableSlot **slots, int ntuples, CommandId cid, int options, BulkInsertState bistate) {
  elog(DEBUG1, "ccam_multi_insert called");
}

static TM_Result ccam_tuple_delete(Relation relation, ItemPointer tid, CommandId cid, Snapshot snapshot, Snapshot crosscheck, bool wait, TM_FailureData *tmfd, bool changingPart) {
  elog(DEBUG1, "ccam_tuple_delete called");
  return TM_Ok;
}

static TM_Result ccam_tuple_update(Relation relation, ItemPointer otid, TupleTableSlot *slot, CommandId cid, Snapshot snapshot, Snapshot crosscheck, bool wait, TM_FailureData *tmfd, LockTupleMode *lockmode, TU_UpdateIndexes *update_indexes) {
  elog(DEBUG1, "ccam_tuple_update called");
  return TM_Ok;
}

static TM_Result ccam_tuple_lock(Relation relation, ItemPointer tid, Snapshot snapshot, TupleTableSlot *slot, CommandId cid, LockTupleMode mode, LockWaitPolicy wait_policy, uint8 flags, TM_FailureData *tmfd) {
  elog(DEBUG1, "ccam_tuple_lock called");
  return TM_Ok;
}

static bool ccam_fetch_row_version(Relation relation, ItemPointer tid, Snapshot snapshot, TupleTableSlot *slot) {
  elog(DEBUG1, "ccam_fetch_row_version called");
  return false;
}

static void ccam_get_latest_tid(TableScanDesc sscan, ItemPointer tid) {
  elog(DEBUG1, "ccam_get_latest_tid called");
}

static bool ccam_tuple_tid_valid(TableScanDesc scan, ItemPointer tid) {
  elog(DEBUG1, "ccam_tuple_tid_valid called");
  return false;
}

static bool ccam_tuple_satisfies_snapshot(Relation rel, TupleTableSlot *slot, Snapshot snapshot) {
  elog(DEBUG1, "ccam_tuple_satisfies_snapshot called");
  return false;
}

static TransactionId ccam_index_delete_tuples(Relation rel, TM_IndexDeleteOp *delstate) {
  elog(DEBUG1, "ccam_index_delete_tuples called");
  return InvalidTransactionId;
}

static void ccam_relation_set_new_filelocator(Relation rel, const RelFileLocator *newrlocator, char persistence, TransactionId *freezeXid, MultiXactId *minmulti) {
  elog(DEBUG1, "ccam_relation_set_new_filelocator called");
}

static void ccam_relation_nontransactional_truncate(Relation rel) {
  elog(DEBUG1, "ccam_relation_nontransactional_truncate called");
}

static void ccam_relation_copy_data(Relation rel, const RelFileLocator *newrlocator) {
  elog(DEBUG1, "ccam_relation_copy_data called");
}

static void ccam_relation_copy_for_cluster(Relation OldHeap, Relation NewHeap, Relation OldIndex, bool use_sort, TransactionId OldestXmin, TransactionId *xid_cutoff, MultiXactId *multi_cutoff, double *num_tuples, double *tups_vacuumed, double *tups_recently_dead) {
  elog(DEBUG1, "ccam_relation_copy_for_cluster called");
}

static void ccam_vacuum_rel(Relation rel, VacuumParams *params, BufferAccessStrategy bstrategy) {
  elog(DEBUG1, "ccam_vacuum_rel called");
}

static bool ccam_scan_analyze_next_block(TableScanDesc scan, BlockNumber blockno, BufferAccessStrategy bstrategy) {
  elog(DEBUG1, "ccam_scan_analyze_next_block called");
  return false;
}

static bool ccam_scan_analyze_next_tuple(TableScanDesc scan, TransactionId OldestXmin, double *liverows, double *deadrows, TupleTableSlot *slot) {
  elog(DEBUG1, "ccam_scan_analyze_next_tuple called");
  return false;
}

static double ccam_index_build_range_scan(Relation heapRelation, Relation indexRelation, IndexInfo *indexInfo, bool allow_sync, bool anyvisible, bool progress, BlockNumber start_blockno, BlockNumber numblocks, IndexBuildCallback callback, void *callback_state, TableScanDesc scan) {
  elog(DEBUG1, "ccam_index_build_range_scan called");
  return 0;
}

static void ccam_index_validate_scan(Relation heapRelation, Relation indexRelation, IndexInfo *indexInfo, Snapshot snapshot, ValidateIndexState *state) {
  elog(DEBUG1, "ccam_index_validate_scan called");
}

static bool ccam_relation_needs_toast_table(Relation rel) {
  elog(DEBUG1, "ccam_relation_needs_toast_table called");
  return false;
}

static Oid ccam_relation_toast_am(Relation rel) {
  elog(DEBUG1, "ccam_relation_toast_am called");
  return InvalidOid;
}

static void ccam_fetch_toast_slice(Relation toastrel, Oid valueid, int32 attrsize, int32 sliceoffset, int32 slicelength, struct varlena *result) {
  elog(DEBUG1, "ccam_fetch_toast_slice called");
}

static void ccam_estimate_rel_size(Relation rel, int32 *attr_widths, BlockNumber *pages, double *tuples, double *allvisfrac) {
  elog(DEBUG1, "ccam_estimate_rel_size called");
}

static bool ccam_scan_sample_next_block(TableScanDesc scan, SampleScanState *scanstate) {
  elog(DEBUG1, "ccam_scan_sample_next_block called");
  return false;
}

static bool ccam_scan_sample_next_tuple(TableScanDesc scan, SampleScanState *scanstate, TupleTableSlot *slot) {
  elog(DEBUG1, "ccam_scan_sample_next_tuple called");
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
  .relation_nontransactional_truncate = ccam_relation_nontransactional_truncate,
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
  .scan_sample_next_tuple = ccam_scan_sample_next_tuple
};

Datum
ccam_tableam_handler(PG_FUNCTION_ARGS)
{
    PG_RETURN_POINTER(&ccam_methods);
}
