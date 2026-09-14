#include "postgres.h"

#include "nr_kernel.h"
#include "access/heapam.h"
#include "access/tableam.h"
#include "access/genam.h"
#include "catalog/nr_aiengine.h"
#include "catalog/indexing.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "executor/executor.h"
#include "tcop/utility.h"
#include "utils/guc.h"
#include "utils/snapmgr.h"
#include "utils/builtins.h"

#include "nodePredict.h"

/* required metadata marker for PostgreSQL extensions */
PG_MODULE_MAGIC;

/*
 * neurdb.predict_into --- optional target table for PREDICT materialization.
 *
 * When this GUC is set to a (pre-created) table name, the PREDICT AI operator
 * emits every input row augmented with a trailing "nr_pred" column AND writes
 * that augmented row into the named table.  This turns PREDICT into a
 * relation-producing operator whose result can be consumed by ordinary SQL
 * (e.g. ORDER BY ... LIMIT k for top-k, or GROUP BY for aggregation),
 * effectively enabling "PREDICT -> top-k / aggregation" pipelines.
 *
 * When empty (the default), PREDICT keeps its legacy single-column output and
 * does not write anywhere, so existing behaviour is preserved.
 */
char	   *NrPredictInto = NULL;

extern planner_hook_type planner_hook;
extern ExecutorStart_hook_type ExecutorStart_hook;
extern ExecutorRun_hook_type ExecutorRun_hook;
extern ExecutorEnd_hook_type ExecutorEnd_hook;
extern ExecutorFinish_hook_type ExecutorFinish_hook;

planner_hook_type original_planner_hook = NULL;
ExecutorStart_hook_type original_executorstart_hook = NULL;
ExecutorRun_hook_type original_executorrun_hook = NULL;
ExecutorEnd_hook_type original_executorend_hook = NULL;
ExecutorFinish_hook_type original_executorfinish_hook = NULL;


static Oid
_assign_oid(Relation rel)
{
	return GetNewOidWithIndex(rel,
							  NrAiengineOidIndexId,
							  Anum_nr_aiengine_oid);
}

static HeapTuple
_build_ai_engine_tuple(const char *addr, int port, Relation rel)
{
	Datum		values[Natts_nr_aiengine];
	bool		nulls[Natts_nr_aiengine];

	for (int i = 0; i < Natts_nr_aiengine; i++)
	{
		nulls[i] = true;
	}

	values[Anum_nr_aiengine_oid - 1] = ObjectIdGetDatum(_assign_oid(rel));
	nulls[Anum_nr_aiengine_oid - 1] = false;

	values[Anum_nr_aiengine_aieaddr - 1] = CStringGetTextDatum(addr);
	nulls[Anum_nr_aiengine_aieaddr - 1] = false;

	values[Anum_nr_aiengine_aieport - 1] = Int32GetDatum(port);
	nulls[Anum_nr_aiengine_aieport - 1] = false;

	return heap_form_tuple(RelationGetDescr(rel), values, nulls);
}


static void
_clear_all_tuples(Relation rel)
{
	HeapTuple	tup;
	SysScanDesc scan;

	scan = table_beginscan_catalog(rel, 0, NULL);

	while ((tup = heap_getnext(scan, ForwardScanDirection)) != NULL)
	{
		CatalogTupleDelete(rel, &tup->t_self);
	}

	table_endscan(scan);
}

void
aiengineworker_main(Datum main_arg)
{
	Relation	rel;
	HeapTuple	tup;

	elog(DEBUG1, "In NeurDB's aiengineworker_main");

	/* Required signal handling */
	pqsignal(SIGTERM, die);
	BackgroundWorkerUnblockSignals();

	/* Initialize connection to neurdb database */
	BackgroundWorkerInitializeConnection("neurdb", NULL, 0);
	StartTransactionCommand();
	PushActiveSnapshot(GetTransactionSnapshot());

	/* Open system catalog nr_aiengine */
	rel = table_open(NrAiengineRelationId, RowExclusiveLock);

	elog(LOG, "opened system catalog nr_aiengine (relid=%u)",
		 RelationGetRelid(rel));

	/* clear all tuples in system catalog nr_aiengine */
	_clear_all_tuples(rel);
	elog(DEBUG1, "cleared all tuples in system catalog nr_aiengine");

	/* add a new tuple into the relation */
	tup = _build_ai_engine_tuple("127.0.0.1", 8090, rel);
	CatalogTupleInsert(rel, tup);
	CommandCounterIncrement();

	table_close(rel, RowExclusiveLock);

	/* Release the snapshot and commit the transaction */
	PopActiveSnapshot();
	CommitTransactionCommand();

	proc_exit(0);
}


void
register_aiengine_background_worker(void)
{
	BackgroundWorker worker;

	if (!process_shared_preload_libraries_in_progress)
		return;

	MemSet(&worker, 0, sizeof(BackgroundWorker));

	worker.bgw_flags =
		BGWORKER_SHMEM_ACCESS |
		BGWORKER_BACKEND_DATABASE_CONNECTION;

	worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
	worker.bgw_restart_time = BGW_NEVER_RESTART;

	snprintf(worker.bgw_name, BGW_MAXLEN, "AI Engine Catalog Manager");
	snprintf(worker.bgw_library_name, BGW_MAXLEN, "nr_ext");
	snprintf(worker.bgw_function_name, BGW_MAXLEN, "aiengineworker_main");

	worker.bgw_main_arg = (Datum) 0;
	worker.bgw_notify_pid = 0;

	RegisterBackgroundWorker(&worker);
}


/*  Called upon extension load. */
void
_PG_init(void)
{
	elog(DEBUG1, "In NeurDB's _PG_init");

	/*
	 * Register the materialization target GUC.  PGC_USERSET so it can be set
	 * per-session/per-statement (e.g. SET neurdb.predict_into = 'w_pred_1';).
	 */
	DefineCustomStringVariable("neurdb.predict_into",
							   "Target table for PREDICT materialization.",
							   "If non-empty, PREDICT appends an nr_pred column to "
							   "each input row and writes the augmented row into "
							   "this (pre-created) table, enabling PREDICT -> "
							   "top-k / aggregation pipelines via plain SQL.",
							   &NrPredictInto,
							   "",
							   PGC_USERSET,
							   0,
							   NULL, NULL, NULL);

	/* Save the original hook value. */
	original_planner_hook = planner_hook;
	original_executorstart_hook = ExecutorStart_hook;
	original_executorrun_hook = ExecutorRun_hook;
	original_executorend_hook = ExecutorEnd_hook;
	original_executorfinish_hook = ExecutorFinish_hook;
	/* Register our handler. */
	planner_hook = NeurDB_planner;
	ExecutorStart_hook = NeurDB_ExecutorStart;
	ExecutorRun_hook = NeurDB_ExecutorRun;
	ExecutorFinish_hook = NeurDB_ExecutorFinish;
	ExecutorEnd_hook = NeurDB_ExecutorEnd;

	/*
	 * Let the core executor dispatch NeurDBPredict nodes that appear nested
	 * inside plan trees (SELECT ... FROM (PREDICT ...) p) back into this
	 * extension's node implementation.
	 */
	NeurDBPredictInitNode_hook = NeurDBPredictInitNodeHook;
	NeurDBPredictEndNode_hook = NeurDBPredictEndNodeHook;
	NeurDBPredictReScan_hook = NeurDBPredictReScanHook;

	register_aiengine_background_worker();
}

/*  Called with extension unload. */
void
_PG_fini(void)
{
	elog(DEBUG1, "In NeurDB's _PG_fini");
	/* Return back the original hook value. */
	planner_hook = original_planner_hook;
	ExecutorStart_hook = original_executorstart_hook;
	ExecutorRun_hook = original_executorrun_hook;
	ExecutorEnd_hook = original_executorend_hook;
	ExecutorFinish_hook = original_executorfinish_hook;
	NeurDBPredictInitNode_hook = NULL;
	NeurDBPredictEndNode_hook = NULL;
	NeurDBPredictReScan_hook = NULL;
}
