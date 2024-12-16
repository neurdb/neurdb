#include "postgres.h"

#include "access/xact.h"
#include "catalog/namespace.h"
#include "commands/trigger.h"
#include "executor/executor.h"
#include "foreign/fdwapi.h"
#include "miscadmin.h"
#include "tcop/utility.h"
#include "utils/lsyscache.h"
#include "utils/snapmgr.h"

PG_MODULE_MAGIC;

extern ExecutorStart_hook_type ExecutorStart_hook;
extern ExecutorRun_hook_type ExecutorRun_hook;

ExecutorStart_hook_type original_executorstart_hook = NULL;
ExecutorRun_hook_type original_executorrun_hook = NULL;

/* decls for local routines only used within this module */
/*  static void InitPlan(QueryDesc *queryDesc, int eflags); */
/*  static void CheckValidRowMarkRel(Relation rel, RowMarkType markType); */
/*  static void ExecPostprocessPlan(EState *estate); */
/*  static void ExecEndPlan(PlanState *planstate, EState *estate); */
/*  static void ExecutePlan(EState *estate, PlanState *planstate, */
/*  						bool use_parallel_mode, */
/*  						CmdType operation, */
/*  						bool sendTuples, */
/*  						uint64 numberTuples, */
/*  						ScanDirection direction, */
/*  						DestReceiver *dest, */
/*  						bool execute_once); */
/*  static bool ExecCheckOneRelPerms(RTEPermissionInfo *perminfo); */
/*  static bool ExecCheckPermissionsModified(Oid relOid, Oid userid, */
/*  										 Bitmapset *modifiedCols, */
/*  										 AclMode requiredPerms); */
/*  static void ExecCheckXactReadOnly(PlannedStmt *plannedstmt); */
/*  static char *ExecBuildSlotValueDescription(Oid reloid, */
/*  										   TupleTableSlot *slot, */
/*  										   TupleDesc tupdesc, */
/*  										   Bitmapset *modifiedCols, */
/*  										   int maxfieldlen); */
/*  static void EvalPlanQualStart(EPQState *epqstate, Plan *planTree); */


/* --- START ---------------------------------------------------------------- */

/*
 * Check that the query does not imply any writes to non-temp tables;
 * unless we're in parallel mode, in which case don't even allow writes
 * to temp tables.
 *
 * Note: in a Hot Standby this would need to reject writes to temp
 * tables just as we do in parallel mode; but an HS standby can't have created
 * any temp tables in the first place, so no need to check that.
 */
static void
ExecCheckXactReadOnly(PlannedStmt *plannedstmt)
{
	ListCell   *l;

	/*
	 * Fail if write permissions are requested in parallel mode for table
	 * (temp or non-temp), otherwise fail for any non-temp table.
	 */
	foreach(l, plannedstmt->permInfos)
	{
		RTEPermissionInfo *perminfo = lfirst_node(RTEPermissionInfo, l);

		if ((perminfo->requiredPerms & (~ACL_SELECT)) == 0)
			continue;

		if (isTempNamespace(get_rel_namespace(perminfo->relid)))
			continue;

		PreventCommandIfReadOnly(CreateCommandName((Node *) plannedstmt));
	}

	if (plannedstmt->commandType != CMD_SELECT || plannedstmt->hasModifyingCTE)
		PreventCommandIfParallelMode(CreateCommandName((Node *) plannedstmt));
}


/*
 * Check that a proposed rowmark target relation is a legal target
 *
 * In most cases parser and/or planner should have noticed this already, but
 * they don't cover all cases.
 */
static void
CheckValidRowMarkRel(Relation rel, RowMarkType markType)
{
	FdwRoutine *fdwroutine;

	switch (rel->rd_rel->relkind)
	{
		case RELKIND_RELATION:
		case RELKIND_PARTITIONED_TABLE:
			/* OK */
			break;
		case RELKIND_SEQUENCE:
			/* Must disallow this because we don't vacuum sequences */
			ereport(ERROR,
					(errcode(ERRCODE_WRONG_OBJECT_TYPE),
					 errmsg("cannot lock rows in sequence \"%s\"",
							RelationGetRelationName(rel))));
			break;
		case RELKIND_TOASTVALUE:
			/* We could allow this, but there seems no good reason to */
			ereport(ERROR,
					(errcode(ERRCODE_WRONG_OBJECT_TYPE),
					 errmsg("cannot lock rows in TOAST relation \"%s\"",
							RelationGetRelationName(rel))));
			break;
		case RELKIND_VIEW:
			/* Should not get here; planner should have expanded the view */
			ereport(ERROR,
					(errcode(ERRCODE_WRONG_OBJECT_TYPE),
					 errmsg("cannot lock rows in view \"%s\"",
							RelationGetRelationName(rel))));
			break;
		case RELKIND_MATVIEW:
			/* Allow referencing a matview, but not actual locking clauses */
			if (markType != ROW_MARK_REFERENCE)
				ereport(ERROR,
						(errcode(ERRCODE_WRONG_OBJECT_TYPE),
						 errmsg("cannot lock rows in materialized view \"%s\"",
								RelationGetRelationName(rel))));
			break;
		case RELKIND_FOREIGN_TABLE:
			/* Okay only if the FDW supports it */
			fdwroutine = GetFdwRoutineForRelation(rel, false);
			if (fdwroutine->RefetchForeignRow == NULL)
				ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("cannot lock rows in foreign table \"%s\"",
								RelationGetRelationName(rel))));
			break;
		default:
			ereport(ERROR,
					(errcode(ERRCODE_WRONG_OBJECT_TYPE),
					 errmsg("cannot lock rows in relation \"%s\"",
							RelationGetRelationName(rel))));
			break;
	}
}

/* ----------------------------------------------------------------
 *		InitPlan
 *
 *		Initializes the query plan: open files, allocate storage
 *		and start up the rule manager
 * ----------------------------------------------------------------
 */
static void
InitPlan(QueryDesc *queryDesc, int eflags)
{
	CmdType		operation = queryDesc->operation;
	PlannedStmt *plannedstmt = queryDesc->plannedstmt;
	Plan	   *plan = plannedstmt->planTree;
	List	   *rangeTable = plannedstmt->rtable;
	EState	   *estate = queryDesc->estate;
	PlanState  *planstate;
	TupleDesc	tupType;
	ListCell   *l;
	int			i;

	/*
	 * Do permissions checks
	 */
	ExecCheckPermissions(rangeTable, plannedstmt->permInfos, true);

	/*
	 * initialize the node's execution state
	 */
	ExecInitRangeTable(estate, rangeTable, plannedstmt->permInfos);

	estate->es_plannedstmt = plannedstmt;

	/*
	 * Next, build the ExecRowMark array from the PlanRowMark(s), if any.
	 */
	if (plannedstmt->rowMarks)
	{
		estate->es_rowmarks = (ExecRowMark **)
			palloc0(estate->es_range_table_size * sizeof(ExecRowMark *));
		foreach(l, plannedstmt->rowMarks)
		{
			PlanRowMark *rc = (PlanRowMark *) lfirst(l);
			Oid			relid;
			Relation	relation;
			ExecRowMark *erm;

			/* ignore "parent" rowmarks; they are irrelevant at runtime */
			if (rc->isParent)
				continue;

			/* get relation's OID (will produce InvalidOid if subquery) */
			relid = exec_rt_fetch(rc->rti, estate)->relid;

			/* open relation, if we need to access it for this mark type */
			switch (rc->markType)
			{
				case ROW_MARK_EXCLUSIVE:
				case ROW_MARK_NOKEYEXCLUSIVE:
				case ROW_MARK_SHARE:
				case ROW_MARK_KEYSHARE:
				case ROW_MARK_REFERENCE:
					relation = ExecGetRangeTableRelation(estate, rc->rti);
					break;
				case ROW_MARK_COPY:
					/* no physical table access is required */
					relation = NULL;
					break;
				default:
					elog(ERROR, "unrecognized markType: %d", rc->markType);
					relation = NULL;	/* keep compiler quiet */
					break;
			}

			/* Check that relation is a legal target for marking */
			if (relation)
				CheckValidRowMarkRel(relation, rc->markType);

			erm = (ExecRowMark *) palloc(sizeof(ExecRowMark));
			erm->relation = relation;
			erm->relid = relid;
			erm->rti = rc->rti;
			erm->prti = rc->prti;
			erm->rowmarkId = rc->rowmarkId;
			erm->markType = rc->markType;
			erm->strength = rc->strength;
			erm->waitPolicy = rc->waitPolicy;
			erm->ermActive = false;
			ItemPointerSetInvalid(&(erm->curCtid));
			erm->ermExtra = NULL;

			Assert(erm->rti > 0 && erm->rti <= estate->es_range_table_size &&
				   estate->es_rowmarks[erm->rti - 1] == NULL);

			estate->es_rowmarks[erm->rti - 1] = erm;
		}
	}

	/*
	 * Initialize the executor's tuple table to empty.
	 */
	estate->es_tupleTable = NIL;

	/* signal that this EState is not used for EPQ */
	estate->es_epq_active = NULL;

	/*
	 * Initialize private state information for each SubPlan.  We must do this
	 * before running ExecInitNode on the main query tree, since
	 * ExecInitSubPlan expects to be able to find these entries.
	 */
	Assert(estate->es_subplanstates == NIL);
	i = 1;						/* subplan indices count from 1 */
	foreach(l, plannedstmt->subplans)
	{
		Plan	   *subplan = (Plan *) lfirst(l);
		PlanState  *subplanstate;
		int			sp_eflags;

		/*
		 * A subplan will never need to do BACKWARD scan nor MARK/RESTORE. If
		 * it is a parameterless subplan (not initplan), we suggest that it be
		 * prepared to handle REWIND efficiently; otherwise there is no need.
		 */
		sp_eflags = eflags
			& ~(EXEC_FLAG_REWIND | EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK);
		if (bms_is_member(i, plannedstmt->rewindPlanIDs))
			sp_eflags |= EXEC_FLAG_REWIND;

		subplanstate = ExecInitNode(subplan, estate, sp_eflags);

		estate->es_subplanstates = lappend(estate->es_subplanstates,
										   subplanstate);

		i++;
	}

	/*
	 * Initialize the private state information for all the nodes in the query
	 * tree.  This opens files, allocates storage and leaves us ready to start
	 * processing tuples.
	 */
	planstate = ExecInitNode(plan, estate, eflags);

	/*
	 * Get the tuple descriptor describing the type of tuples to return.
	 */
	tupType = ExecGetResultType(planstate);

	/*
	 * Initialize the junk filter if needed.  SELECT queries need a filter if
	 * there are any junk attrs in the top-level tlist.
	 */
	if (operation == CMD_SELECT)
	{
		bool		junk_filter_needed = false;
		ListCell   *tlist;

		foreach(tlist, plan->targetlist)
		{
			TargetEntry *tle = (TargetEntry *) lfirst(tlist);

			if (tle->resjunk)
			{
				junk_filter_needed = true;
				break;
			}
		}

		if (junk_filter_needed)
		{
			JunkFilter *j;
			TupleTableSlot *slot;

			slot = ExecInitExtraTupleSlot(estate, NULL, &TTSOpsVirtual);
			j = ExecInitJunkFilter(planstate->plan->targetlist,
								   slot);
			estate->es_junkFilter = j;

			/* Want to return the cleaned tuple type */
			tupType = j->jf_cleanTupType;
		}
	}

	queryDesc->tupDesc = tupType;
	queryDesc->planstate = planstate;
}


void
NeurDB_ExecutorStart(QueryDesc *queryDesc, int eflags)
{
	elog(DEBUG1, "In NeurDB's executor start");

	EState	   *estate;
	MemoryContext oldcontext;

	/* sanity checks: queryDesc must not be started already */
	Assert(queryDesc != NULL);
	Assert(queryDesc->estate == NULL);

	/*
	 * If the transaction is read-only, we need to check if any writes are
	 * planned to non-temporary tables.  EXPLAIN is considered read-only.
	 *
	 * Don't allow writes in parallel mode.  Supporting UPDATE and DELETE
	 * would require (a) storing the combo CID hash in shared memory, rather
	 * than synchronizing it just once at the start of parallelism, and (b) an
	 * alternative to heap_update()'s reliance on xmax for mutual exclusion.
	 * INSERT may have no such troubles, but we forbid it to simplify the
	 * checks.
	 *
	 * We have lower-level defenses in CommandCounterIncrement and elsewhere
	 * against performing unsafe operations in parallel mode, but this gives a
	 * more user-friendly error message.
	 */
	if ((XactReadOnly || IsInParallelMode()) &&
		!(eflags & EXEC_FLAG_EXPLAIN_ONLY))
		ExecCheckXactReadOnly(queryDesc->plannedstmt);

	/*
	 * Build EState, switch into per-query memory context for startup.
	 */
	estate = CreateExecutorState();
	queryDesc->estate = estate;

	oldcontext = MemoryContextSwitchTo(estate->es_query_cxt);

	/*
	 * Fill in external parameters, if any, from queryDesc; and allocate
	 * workspace for internal parameters
	 */
	estate->es_param_list_info = queryDesc->params;

	if (queryDesc->plannedstmt->paramExecTypes != NIL)
	{
		int			nParamExec;

		nParamExec = list_length(queryDesc->plannedstmt->paramExecTypes);
		estate->es_param_exec_vals = (ParamExecData *)
			palloc0(nParamExec * sizeof(ParamExecData));
	}

	/* We now require all callers to provide sourceText */
	Assert(queryDesc->sourceText != NULL);
	estate->es_sourceText = queryDesc->sourceText;

	/*
	 * Fill in the query environment, if any, from queryDesc.
	 */
	estate->es_queryEnv = queryDesc->queryEnv;

	/*
	 * If non-read-only query, set the command ID to mark output tuples with
	 */
	switch (queryDesc->operation)
	{
		case CMD_SELECT:

			/*
			 * SELECT FOR [KEY] UPDATE/SHARE and modifying CTEs need to mark
			 * tuples
			 */
			if (queryDesc->plannedstmt->rowMarks != NIL ||
				queryDesc->plannedstmt->hasModifyingCTE)
				estate->es_output_cid = GetCurrentCommandId(true);

			/*
			 * A SELECT without modifying CTEs can't possibly queue triggers,
			 * so force skip-triggers mode. This is just a marginal efficiency
			 * hack, since AfterTriggerBeginQuery/AfterTriggerEndQuery aren't
			 * all that expensive, but we might as well do it.
			 */
			if (!queryDesc->plannedstmt->hasModifyingCTE)
				eflags |= EXEC_FLAG_SKIP_TRIGGERS;
			break;

		case CMD_INSERT:
		case CMD_DELETE:
		case CMD_UPDATE:
		case CMD_MERGE:
			estate->es_output_cid = GetCurrentCommandId(true);
			break;

		default:
			elog(ERROR, "unrecognized operation code: %d",
				 (int) queryDesc->operation);
			break;
	}

	/*
	 * Copy other important information into the EState
	 */
	estate->es_snapshot = RegisterSnapshot(queryDesc->snapshot);
	estate->es_crosscheck_snapshot = RegisterSnapshot(queryDesc->crosscheck_snapshot);
	estate->es_top_eflags = eflags;
	estate->es_instrument = queryDesc->instrument_options;
	estate->es_jit_flags = queryDesc->plannedstmt->jitFlags;

	/*
	 * Set up an AFTER-trigger statement context, unless told not to, or
	 * unless it's EXPLAIN-only mode (when ExecutorFinish won't be called).
	 */
	if (!(eflags & (EXEC_FLAG_SKIP_TRIGGERS | EXEC_FLAG_EXPLAIN_ONLY)))
		AfterTriggerBeginQuery();

	/*
	 * Initialize the plan state tree
	 */
	InitPlan(queryDesc, eflags);

	MemoryContextSwitchTo(oldcontext);
}

/* --- RUN ------------------------------------------------------------------ */

/* ----------------------------------------------------------------
 *		ExecutePlan
 *
 *		Processes the query plan until we have retrieved 'numberTuples' tuples,
 *		moving in the specified direction.
 *
 *		Runs to completion if numberTuples is 0
 *
 * Note: the ctid attribute is a 'junk' attribute that is removed before the
 * user can see it
 * ----------------------------------------------------------------
 */
static void
ExecutePlan(EState *estate,
			PlanState *planstate,
			bool use_parallel_mode,
			CmdType operation,
			bool sendTuples,
			uint64 numberTuples,
			ScanDirection direction,
			DestReceiver *dest,
			bool execute_once)
{
	TupleTableSlot *slot;
	uint64		current_tuple_count;

	/*
	 * initialize local variables
	 */
	current_tuple_count = 0;

	/*
	 * Set the direction.
	 */
	estate->es_direction = direction;

	/*
	 * If the plan might potentially be executed multiple times, we must force
	 * it to run without parallelism, because we might exit early.
	 */
	if (!execute_once)
		use_parallel_mode = false;

	estate->es_use_parallel_mode = use_parallel_mode;
	if (use_parallel_mode)
		EnterParallelMode();

	/*
	 * Loop until we've processed the proper number of tuples from the plan.
	 */
	for (;;)
	{
		/* Reset the per-output-tuple exprcontext */
		ResetPerTupleExprContext(estate);

		/*
		 * Execute the plan and obtain a tuple
		 */
		slot = ExecProcNode(planstate);

		/*
		 * if the tuple is null, then we assume there is nothing more to
		 * process so we just end the loop...
		 */
		if (TupIsNull(slot))
			break;

		/*
		 * If we have a junk filter, then project a new tuple with the junk
		 * removed.
		 *
		 * Store this new "clean" tuple in the junkfilter's resultSlot.
		 * (Formerly, we stored it back over the "dirty" tuple, which is WRONG
		 * because that tuple slot has the wrong descriptor.)
		 */
		if (estate->es_junkFilter != NULL)
			slot = ExecFilterJunk(estate->es_junkFilter, slot);

		/*
		 * If we are supposed to send the tuple somewhere, do so. (In
		 * practice, this is probably always the case at this point.)
		 */
		if (sendTuples)
		{
			/*
			 * If we are not able to send the tuple, we assume the destination
			 * has closed and no more tuples can be sent. If that's the case,
			 * end the loop.
			 */
			if (!dest->receiveSlot(slot, dest))
				break;
		}

		/*
		 * Count tuples processed, if this is a SELECT.  (For other operation
		 * types, the ModifyTable plan node must count the appropriate
		 * events.)
		 */
		if (operation == CMD_SELECT)
			(estate->es_processed)++;

		/*
		 * check our tuple count.. if we've processed the proper number then
		 * quit, else loop again and process more tuples.  Zero numberTuples
		 * means no limit.
		 */
		current_tuple_count++;
		if (numberTuples && numberTuples == current_tuple_count)
			break;
	}

	/*
	 * If we know we won't need to back up, we can release resources at this
	 * point.
	 */
	if (!(estate->es_top_eflags & EXEC_FLAG_BACKWARD))
		ExecShutdownNode(planstate);

	if (use_parallel_mode)
		ExitParallelMode();
}




void
NeurDB_ExecutorRun(QueryDesc *queryDesc, ScanDirection direction, uint64 count, bool execute_once)
{
	elog(DEBUG1, "In NeurDB's executor run");

	EState	   *estate;
	CmdType		operation;
	DestReceiver *dest;
	bool		sendTuples;
	MemoryContext oldcontext;

	/* sanity checks */
	Assert(queryDesc != NULL);

	estate = queryDesc->estate;

	Assert(estate != NULL);
	Assert(!(estate->es_top_eflags & EXEC_FLAG_EXPLAIN_ONLY));

	/*
	 * Switch into per-query memory context
	 */
	oldcontext = MemoryContextSwitchTo(estate->es_query_cxt);

	/* Allow instrumentation of Executor overall runtime */
	if (queryDesc->totaltime)
		InstrStartNode(queryDesc->totaltime);

	/*
	 * extract information from the query descriptor and the query feature.
	 */
	operation = queryDesc->operation;
	dest = queryDesc->dest;

	/*
	 * startup tuple receiver, if we will be emitting tuples
	 */
	estate->es_processed = 0;

	sendTuples = (operation == CMD_SELECT ||
				  queryDesc->plannedstmt->hasReturning);

	if (sendTuples)
		dest->rStartup(dest, operation, queryDesc->tupDesc);

	/*
	 * run plan
	 */
	if (!ScanDirectionIsNoMovement(direction))
	{
		if (execute_once && queryDesc->already_executed)
			elog(ERROR, "can't re-execute query flagged for single execution");
		queryDesc->already_executed = true;

		ExecutePlan(estate,
					queryDesc->planstate,
					queryDesc->plannedstmt->parallelModeNeeded,
					operation,
					sendTuples,
					count,
					direction,
					dest,
					execute_once);
	}

	/*
	 * Update es_total_processed to keep track of the number of tuples
	 * processed across multiple ExecutorRun() calls.
	 */
	estate->es_total_processed += estate->es_processed;

	/*
	 * shutdown tuple receiver, if we started it
	 */
	if (sendTuples)
		dest->rShutdown(dest);

	if (queryDesc->totaltime)
		InstrStopNode(queryDesc->totaltime, estate->es_processed);

	MemoryContextSwitchTo(oldcontext);
}


/*  Called upon extension load. */
void
_PG_init(void)
{
	/* Save the original hook value. */
	original_executorstart_hook = ExecutorStart_hook;
	original_executorrun_hook = ExecutorRun_hook;
	/* Register our handler. */
	ExecutorStart_hook = NeurDB_ExecutorStart;
	ExecutorRun_hook = NeurDB_ExecutorRun;
}

/*  Called with extension unload. */
void
_PG_fini(void)
{
	/* Return back the original hook value. */
	ExecutorStart_hook = original_executorstart_hook;
	ExecutorRun_hook = original_executorrun_hook;
}
