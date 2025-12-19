#include "postgres.h"

#include "nr_kernel.h"

/* required metadata marker for PostgreSQL extensions */
PG_MODULE_MAGIC;

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

/*  Called upon extension load. */
void
_PG_init(void)
{
	elog(DEBUG1, "In NeurDB's _PG_init");
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
}
