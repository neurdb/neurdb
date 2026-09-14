/*-------------------------------------------------------------------------
 *
 * nodeNqoAdaptiveJoin.c
 *	  Runtime HashJoin-to-NestLoop selection for NQO.
 *
 * The planner supplies a normal HashJoin subtree, an equivalent NestLoop
 * subtree.  The first HashJoin call builds its real hash table; we read the
 * resulting cardinality and either continue that executor state or switch to
 * the equivalent NestLoop subtree.  The probe is deliberately restricted to
 * unparameterized, nonparallel SELECT plans without subplans.
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "executor/nodeNqoAdaptiveJoin.h"

#include "commands/explain.h"
#include "executor/executor.h"
#include "executor/hashjoin.h"
#include "executor/nodeHashjoin.h"
#include "nodes/extensible.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "portability/instr_time.h"

#define NQO_ADAPTIVE_JOIN_NAME "NqoAdaptiveJoin"

static NqoAdaptiveJoinStats nqo_adaptive_stats;

void
nqo_reset_adaptive_join_stats(void)
{
	memset(&nqo_adaptive_stats, 0, sizeof(nqo_adaptive_stats));
}

NqoAdaptiveJoinStats
nqo_get_adaptive_join_stats(void)
{
	return nqo_adaptive_stats;
}

typedef struct NqoAdaptiveJoinState
{
	CustomScanState css;
	PlanState  *hash_state;
	PlanState  *nest_state;
	PlanState  *chosen_state;
	const char *level;
	int			threshold_rows;
	double		estimated_build_rows;
	uint64		run_id;
	int			round;
	int			join_id;
	uint64		actual_build_rows;
	bool		indexed_probe;
	bool		decided;
	bool		use_nestloop;
	bool		join_empty;
} NqoAdaptiveJoinState;

static Node *nqo_create_adaptive_join_state(CustomScan *cscan);
static void nqo_begin_adaptive_join(CustomScanState *node,
									   EState *estate, int eflags);
static TupleTableSlot *nqo_exec_adaptive_join(CustomScanState *node);
static void nqo_end_adaptive_join(CustomScanState *node);
static void nqo_rescan_adaptive_join(CustomScanState *node);
static void nqo_explain_adaptive_join(CustomScanState *node,
										 List *ancestors,
										 ExplainState *es);

static const CustomScanMethods nqo_adaptive_join_plan_methods = {
	.CustomName = NQO_ADAPTIVE_JOIN_NAME,
	.CreateCustomScanState = nqo_create_adaptive_join_state,
};

static const CustomExecMethods nqo_adaptive_join_exec_methods = {
	.CustomName = NQO_ADAPTIVE_JOIN_NAME,
	.BeginCustomScan = nqo_begin_adaptive_join,
	.ExecCustomScan = nqo_exec_adaptive_join,
	.EndCustomScan = nqo_end_adaptive_join,
	.ReScanCustomScan = nqo_rescan_adaptive_join,
	.ExplainCustomScan = nqo_explain_adaptive_join,
};

static void
nqo_register_adaptive_join(void)
{
	if (GetCustomScanMethods(NQO_ADAPTIVE_JOIN_NAME, true) == NULL)
		RegisterCustomScanMethods(&nqo_adaptive_join_plan_methods);
}

static Node *
nqo_create_adaptive_join_state(CustomScan *cscan)
{
	NqoAdaptiveJoinState *state;

	state = palloc0(sizeof(NqoAdaptiveJoinState));
	NodeSetTag(&state->css, T_CustomScanState);
	state->css.methods = &nqo_adaptive_join_exec_methods;
	return (Node *) state;
}

static void
nqo_begin_adaptive_join(CustomScanState *node, EState *estate, int eflags)
{
	NqoAdaptiveJoinState *state = (NqoAdaptiveJoinState *) node;
	CustomScan *cscan = (CustomScan *) node->ss.ps.plan;
	ListCell   *lc;

	Assert(list_length(cscan->custom_plans) == 2);
	Assert(list_length(cscan->custom_private) == 7);

	foreach(lc, cscan->custom_plans)
		node->custom_ps = lappend(node->custom_ps,
								 ExecInitNode((Plan *) lfirst(lc),
											  estate, eflags));

	state->hash_state = (PlanState *) list_nth(node->custom_ps, 0);
	state->nest_state = (PlanState *) list_nth(node->custom_ps, 1);
	state->level = strVal(list_nth(cscan->custom_private, 0));
	state->threshold_rows = intVal(list_nth(cscan->custom_private, 1));
	state->estimated_build_rows =
		strtod(strVal(list_nth(cscan->custom_private, 2)), NULL);
	state->run_id =
		(uint64) strtou64(strVal(list_nth(cscan->custom_private, 3)), NULL, 10);
	state->round = intVal(list_nth(cscan->custom_private, 4));
	state->join_id = intVal(list_nth(cscan->custom_private, 5));
	state->indexed_probe = intVal(list_nth(cscan->custom_private, 6)) != 0;
}

static TupleTableSlot *
nqo_exec_adaptive_join(CustomScanState *node)
{
	NqoAdaptiveJoinState *state = (NqoAdaptiveJoinState *) node;

	if (!state->decided)
	{
		HashJoinState *hashjoin_state;
		instr_time	start;
		instr_time	end;
		bool		build_ok;

		Assert(IsA(state->hash_state, HashJoinState));
		hashjoin_state = (HashJoinState *) state->hash_state;
		INSTR_TIME_SET_CURRENT(start);
		build_ok = ExecHashJoinBuildHashTable(hashjoin_state,
											 &state->actual_build_rows,
											 &state->join_empty);
		INSTR_TIME_SET_CURRENT(end);
		INSTR_TIME_SUBTRACT(end, start);

		if (!build_ok)
			elog(ERROR, "NQO AJoin could not build the HashJoin table");
		state->use_nestloop =
			!state->join_empty &&
			state->actual_build_rows <= (uint64) state->threshold_rows;
		state->chosen_state = state->use_nestloop ?
			state->nest_state : state->hash_state;
		state->decided = true;
		nqo_adaptive_stats.joins_decided++;
		nqo_adaptive_stats.actual_build_rows += state->actual_build_rows;
		nqo_adaptive_stats.build_ms += INSTR_TIME_GET_MILLISEC(end);
		if (state->use_nestloop)
			nqo_adaptive_stats.nestloop_selected++;
		else
			nqo_adaptive_stats.hashjoin_selected++;

		elog(LOG, "[nqo] run=" UINT64_FORMAT
			 " round %d: adaptive_join=%d level=%s "
			 "estimated_build_rows=%.0f actual_build_rows=" UINT64_FORMAT
			 " threshold_rows=%d indexed_probe=%d selected=%s "
			 "hash_build_reused=%d join_empty=%d",
			 state->run_id, state->round, state->join_id, state->level,
			 state->estimated_build_rows, state->actual_build_rows,
			 state->threshold_rows, state->indexed_probe ? 1 : 0,
			 state->use_nestloop ? "NestLoop" : "HashJoin",
			 state->use_nestloop ? 0 : 1,
			 state->join_empty ? 1 : 0);
	}

	if (state->join_empty)
		return NULL;
	return ExecProcNode(state->chosen_state);
}

static void
nqo_end_adaptive_join(CustomScanState *node)
{
	ListCell   *lc;

	foreach(lc, node->custom_ps)
		ExecEndNode((PlanState *) lfirst(lc));
}

static void
nqo_rescan_adaptive_join(CustomScanState *node)
{
	NqoAdaptiveJoinState *state = (NqoAdaptiveJoinState *) node;

	if (state->decided)
		ExecReScan(state->chosen_state);
	else
		ExecReScan(state->hash_state);
}

static void
nqo_explain_adaptive_join(CustomScanState *node, List *ancestors,
							 ExplainState *es)
{
	NqoAdaptiveJoinState *state = (NqoAdaptiveJoinState *) node;

	(void) ancestors;
	ExplainPropertyText("AJoin Action", state->level, es);
	ExplainPropertyInteger("AJoin Switch Threshold", "rows",
						   state->threshold_rows, es);
	ExplainPropertyFloat("Estimated Build Rows", NULL,
						 state->estimated_build_rows, 0, es);
	ExplainPropertyBool("Parameterized Index Probe", state->indexed_probe, es);
	if (state->decided)
	{
		ExplainPropertyUInteger("Observed Build Rows", NULL,
								state->actual_build_rows, es);
		ExplainPropertyText("Selected Join",
							state->use_nestloop ? "NestLoop" : "HashJoin",
							es);
	}
	else
		ExplainPropertyText("Selected Join", "pending execution", es);
}

static Index
nqo_scan_relid(Plan *plan)
{
	switch (nodeTag(plan))
	{
		case T_SeqScan:
		case T_SampleScan:
		case T_IndexScan:
		case T_IndexOnlyScan:
		case T_BitmapIndexScan:
		case T_BitmapHeapScan:
		case T_TidScan:
		case T_TidRangeScan:
		case T_SubqueryScan:
		case T_FunctionScan:
		case T_TableFuncScan:
		case T_ValuesScan:
		case T_CteScan:
		case T_NamedTuplestoreScan:
		case T_WorkTableScan:
		case T_ForeignScan:
			return ((Scan *) plan)->scanrelid;
		default:
			return 0;
	}
}

static Bitmapset *
nqo_collect_plan_relids(Plan *plan)
{
	Bitmapset  *relids = NULL;
	Index		scanrelid;

	if (plan == NULL)
		return NULL;

	scanrelid = nqo_scan_relid(plan);
	if (scanrelid > 0)
		relids = bms_add_member(relids, scanrelid);
	if (IsA(plan, CustomScan))
		relids = bms_join(relids,
						  bms_copy(((CustomScan *) plan)->custom_relids));

	relids = bms_join(relids, nqo_collect_plan_relids(outerPlan(plan)));
	relids = bms_join(relids, nqo_collect_plan_relids(innerPlan(plan)));
	return relids;
}

static bool
nqo_plan_contains_index(Plan *plan)
{
	if (plan == NULL)
		return false;
	if (IsA(plan, IndexScan) || IsA(plan, IndexOnlyScan))
		return true;
	return nqo_plan_contains_index(outerPlan(plan)) ||
		nqo_plan_contains_index(innerPlan(plan));
}

static TargetEntry *
nqo_tle_by_resno(List *targetlist, AttrNumber resno)
{
	ListCell   *lc;

	foreach(lc, targetlist)
	{
		TargetEntry *tle = (TargetEntry *) lfirst(lc);

		if (tle->resno == resno)
			return tle;
	}
	return NULL;
}

static bool
nqo_resolve_output_var(Plan *plan, AttrNumber resno, Index *varno,
						  AttrNumber *varattno, int depth)
{
	TargetEntry *tle;
	Var		   *var;

	if (plan == NULL || resno <= 0 || depth > 32)
		return false;

	tle = nqo_tle_by_resno(plan->targetlist, resno);
	if (tle == NULL || !IsA(tle->expr, Var))
		return false;

	var = (Var *) tle->expr;
	if (var->varattno <= 0)
		return false;
	if (var->varno == OUTER_VAR)
		return nqo_resolve_output_var(outerPlan(plan), var->varattno,
										 varno, varattno, depth + 1);
	if (var->varno == INNER_VAR)
		return nqo_resolve_output_var(innerPlan(plan), var->varattno,
										 varno, varattno, depth + 1);
	if (var->varno == INDEX_VAR && IsA(plan, CustomScan))
	{
		CustomScan *cscan = (CustomScan *) plan;

			if (cscan->methods == &nqo_adaptive_join_plan_methods &&
				cscan->custom_plans != NIL)
				return nqo_resolve_output_var(
					(Plan *) linitial(cscan->custom_plans),
					var->varattno, varno, varattno, depth + 1);
	}
	if (var->varno == INDEX_VAR)
		return false;

	*varno = var->varno;
	*varattno = var->varattno;
	return true;
}

static bool
nqo_find_output_resno(Plan *plan, Index varno, AttrNumber varattno,
						 AttrNumber *resno)
{
	ListCell   *lc;

	foreach(lc, plan->targetlist)
	{
		TargetEntry *tle = (TargetEntry *) lfirst(lc);
		Index		output_varno;
		AttrNumber output_attno;

		if (nqo_resolve_output_var(plan, tle->resno, &output_varno,
									  &output_attno, 0) &&
			output_varno == varno && output_attno == varattno)
		{
			*resno = tle->resno;
			return true;
		}
	}

	return false;
}

static bool
nqo_projection_compatible(Plan *desired_plan, Plan *source_plan)
{
	ListCell   *lc;

	foreach(lc, desired_plan->targetlist)
	{
		TargetEntry *desired_tle = (TargetEntry *) lfirst(lc);
		TargetEntry *source_tle;
		Index		varno;
		AttrNumber varattno;
		AttrNumber source_resno;

		if (!nqo_resolve_output_var(desired_plan,
									   desired_tle->resno,
									   &varno, &varattno, 0) ||
			!nqo_find_output_resno(source_plan, varno, varattno,
									  &source_resno))
			return false;

		source_tle =
			nqo_tle_by_resno(source_plan->targetlist, source_resno);
		Assert(source_tle != NULL);
		if (exprType((Node *) desired_tle->expr) !=
			exprType((Node *) source_tle->expr) ||
			exprTypmod((Node *) desired_tle->expr) !=
			exprTypmod((Node *) source_tle->expr) ||
			exprCollation((Node *) desired_tle->expr) !=
			exprCollation((Node *) source_tle->expr) ||
			desired_tle->resjunk != source_tle->resjunk)
			return false;
	}

	return true;
}

static Plan *
nqo_make_projection(Plan *desired_plan, Plan *source_plan)
{
	Result	   *result = makeNode(Result);
	Plan	   *plan = &result->plan;
	List	   *targetlist = NIL;
	ListCell   *lc;

	/*
	 * setrefs can order the same base columns differently in equivalent
	 * physical plans.  Preserve the desired slot layout while sourcing each
	 * column from the equivalent source output.
	 */
	foreach(lc, desired_plan->targetlist)
	{
		TargetEntry *desired_tle = (TargetEntry *) lfirst(lc);
		TargetEntry *target_tle;
		Index		varno;
		AttrNumber varattno;
		AttrNumber source_resno;
		Var		   *var;

		if (!nqo_resolve_output_var(desired_plan,
									   desired_tle->resno,
									   &varno, &varattno, 0) ||
			!nqo_find_output_resno(source_plan, varno, varattno,
									  &source_resno))
			return NULL;

		var = makeVar(OUTER_VAR, source_resno,
					  exprType((Node *) desired_tle->expr),
					  exprTypmod((Node *) desired_tle->expr),
					  exprCollation((Node *) desired_tle->expr), 0);
		target_tle = makeTargetEntry((Expr *) var, desired_tle->resno,
									desired_tle->resname ?
									pstrdup(desired_tle->resname) : NULL,
									desired_tle->resjunk);
		target_tle->ressortgroupref = desired_tle->ressortgroupref;
		targetlist = lappend(targetlist, target_tle);
	}

	plan->startup_cost = source_plan->startup_cost;
	plan->total_cost = source_plan->total_cost;
	plan->plan_rows = source_plan->plan_rows;
	plan->plan_width = desired_plan->plan_width;
	plan->parallel_aware = false;
	plan->parallel_safe = false;
	plan->async_capable = false;
	plan->plan_node_id = desired_plan->plan_node_id;
	plan->targetlist = targetlist;
	plan->qual = NIL;
	outerPlan(plan) = copyObjectImpl(source_plan);
	innerPlan(plan) = NULL;
	plan->initPlan = NIL;
	plan->extParam = bms_copy(source_plan->extParam);
	plan->allParam = bms_copy(source_plan->allParam);
	result->resconstantqual = NULL;
	return plan;
}

static NestLoop *
nqo_find_nestloop_candidate(Plan *plan, Bitmapset *join_relids,
							   Bitmapset *build_relids,
							   JoinType jointype, Plan *output_plan,
							   Plan *build_plan,
							   bool require_index)
{
	NestLoop   *candidate;
	Bitmapset  *candidate_relids;
	Bitmapset  *outer_relids;
	Bitmapset  *inner_relids;

	if (plan == NULL)
		return NULL;

	if (IsA(plan, NestLoop))
	{
		bool		same_join;
		bool		same_build;
			bool		single_probe;
			bool		same_output;
			bool		index_compatible;

		candidate = (NestLoop *) plan;
		candidate_relids = nqo_collect_plan_relids(plan);
		outer_relids = nqo_collect_plan_relids(outerPlan(plan));
		inner_relids = nqo_collect_plan_relids(innerPlan(plan));
		same_join = candidate->join.jointype == jointype &&
			bms_equal(candidate_relids, join_relids);
		same_build = bms_equal(outer_relids, build_relids);
			single_probe = !require_index ||
				bms_num_members(inner_relids) == 1;
			same_output =
				nqo_projection_compatible(output_plan, plan);
			index_compatible = !require_index ||
				(candidate->nestParams != NIL &&
				 nqo_plan_contains_index(innerPlan(plan)));
				if (same_join)
					elog(DEBUG1, "[nqo] adaptive candidate match: "
						 "same_build=%d single_probe=%d same_output=%d "
						 "index_compatible=%d require_index=%d "
						 "join_relids=%s build_relids=%s "
						 "outer_relids=%s inner_relids=%s",
						 same_build ? 1 : 0, single_probe ? 1 : 0,
						 same_output ? 1 : 0,
						 index_compatible ? 1 : 0, require_index ? 1 : 0,
						 bmsToString(join_relids),
						 bmsToString(build_relids),
						 bmsToString(outer_relids),
						 bmsToString(inner_relids));
				if (same_join && same_build && single_probe && same_output &&
					index_compatible)
		{
			bms_free(candidate_relids);
			bms_free(outer_relids);
			bms_free(inner_relids);
			return candidate;
		}
		bms_free(candidate_relids);
		bms_free(outer_relids);
		bms_free(inner_relids);
	}

	candidate = nqo_find_nestloop_candidate(outerPlan(plan), join_relids,
											   build_relids, jointype,
											   output_plan, build_plan,
											   require_index);
	if (candidate != NULL)
		return candidate;
	return nqo_find_nestloop_candidate(innerPlan(plan), join_relids,
										  build_relids, jointype,
										  output_plan, build_plan,
										  require_index);
}

static List *
nqo_make_passthrough_tlist(List *source)
{
	List	   *targetlist = NIL;
	ListCell   *lc;

	foreach(lc, source)
	{
		TargetEntry *source_tle = (TargetEntry *) lfirst(lc);
		Var		   *var;
		TargetEntry *target_tle;

		var = makeVar(INDEX_VAR, source_tle->resno,
					  exprType((Node *) source_tle->expr),
					  exprTypmod((Node *) source_tle->expr),
					  exprCollation((Node *) source_tle->expr), 0);
		target_tle = makeTargetEntry((Expr *) var, source_tle->resno,
									source_tle->resname ?
									pstrdup(source_tle->resname) : NULL,
									source_tle->resjunk);
		target_tle->ressortgroupref = source_tle->ressortgroupref;
		targetlist = lappend(targetlist, target_tle);
	}
	return targetlist;
}

static CustomScan *
nqo_make_adaptive_join(Plan *hash_plan, NestLoop *nestloop_plan,
						  Plan *build_plan, Bitmapset *join_relids,
						  const char *level, int threshold_rows,
						  bool indexed_probe, uint64 run_id, int round,
						  int join_id)
{
	CustomScan *cscan = makeNode(CustomScan);
	Plan	   *plan = &cscan->scan.plan;
	Plan	   *nestloop_copy = copyObjectImpl(nestloop_plan);
	Plan	   *nestloop_projection;
	int			effective_threshold = indexed_probe ? threshold_rows : 1;

	nestloop_projection = nqo_make_projection(hash_plan, nestloop_copy);
	Assert(nestloop_projection != NULL);
	plan->startup_cost = hash_plan->startup_cost;
	plan->total_cost = hash_plan->total_cost;
	plan->plan_rows = hash_plan->plan_rows;
	plan->plan_width = hash_plan->plan_width;
	plan->parallel_aware = false;
	plan->parallel_safe = false;
	plan->async_capable = false;
	plan->plan_node_id = hash_plan->plan_node_id;
	plan->targetlist = nqo_make_passthrough_tlist(hash_plan->targetlist);
	plan->qual = NIL;
	plan->lefttree = NULL;
	plan->righttree = NULL;
	plan->initPlan = NIL;
	plan->extParam = NULL;
	plan->allParam = NULL;

	cscan->scan.scanrelid = 0;
	cscan->flags = 0;
	cscan->custom_plans = list_make2(hash_plan, nestloop_projection);
	cscan->custom_exprs = NIL;
	cscan->custom_private =
		list_make5(makeString(pstrdup(level)),
				   makeInteger(effective_threshold),
				   makeString(psprintf("%.17g", build_plan->plan_rows)),
				   makeString(psprintf(UINT64_FORMAT, run_id)),
				   makeInteger(round));
	cscan->custom_private =
		lappend(cscan->custom_private, makeInteger(join_id));
	cscan->custom_private =
		lappend(cscan->custom_private,
				makeInteger(indexed_probe ? 1 : 0));
	cscan->custom_scan_tlist = copyObjectImpl(hash_plan->targetlist);
	cscan->custom_relids = bms_copy(join_relids);
	cscan->methods = &nqo_adaptive_join_plan_methods;
	return cscan;
}

static Plan *
nqo_wrap_plan_tree(Plan *plan, Plan *nestloop_root, const char *level,
					  int threshold_rows,
					  int max_nestloop_cost_ratio_pct,
					  uint64 run_id, int round,
					  int *next_join_id, int *wrapped)
{
	HashJoin   *hashjoin;
	Plan	   *hash_node;
	Plan	   *build_plan;
	Bitmapset  *join_relids;
	Bitmapset  *build_relids;
	NestLoop   *candidate;
	bool		indexed_probe = true;

	if (plan == NULL)
		return NULL;

	outerPlan(plan) = nqo_wrap_plan_tree(outerPlan(plan), nestloop_root,
											level, threshold_rows,
											max_nestloop_cost_ratio_pct,
											run_id, round, next_join_id,
											wrapped);
	innerPlan(plan) = nqo_wrap_plan_tree(innerPlan(plan), nestloop_root,
											level, threshold_rows,
											max_nestloop_cost_ratio_pct,
											run_id, round, next_join_id,
											wrapped);

	if (!IsA(plan, HashJoin))
		return plan;

	hashjoin = (HashJoin *) plan;
	hash_node = innerPlan(plan);
	if (hashjoin->join.jointype != JOIN_INNER ||
		hash_node == NULL || !IsA(hash_node, Hash) ||
		plan->parallel_aware ||
		plan->extParam != NULL || plan->initPlan != NIL)
		return plan;

	build_plan = outerPlan(hash_node);
	if (build_plan == NULL)
		return plan;

	join_relids = nqo_collect_plan_relids(plan);
	build_relids = nqo_collect_plan_relids(build_plan);
	candidate = nqo_find_nestloop_candidate(nestloop_root, join_relids,
											  build_relids,
											  hashjoin->join.jointype,
											  plan,
											  build_plan, true);
	if (candidate == NULL)
	{
		indexed_probe = false;
		candidate = nqo_find_nestloop_candidate(nestloop_root, join_relids,
												  build_relids,
												  hashjoin->join.jointype,
												  plan,
												  build_plan,
												  false);
	}

	if (candidate != NULL && !candidate->join.plan.parallel_aware &&
		candidate->join.plan.extParam == NULL &&
		candidate->join.plan.initPlan == NIL)
	{
		CustomScan *adaptive;
		int			join_id = (*next_join_id)++;
		double		hash_cost = Max(plan->total_cost, 0.01);
		double		nestloop_cost = candidate->join.plan.total_cost;
		double		cost_ratio_pct = 100.0 * nestloop_cost / hash_cost;

		elog(LOG, "[nqo] run=" UINT64_FORMAT
			 " round %d: adaptive candidate join=%d indexed_probe=%d "
			 "hash_cost=%.2f nestloop_cost=%.2f cost_ratio_pct=%.2f "
			 "max_cost_ratio_pct=%d",
			 run_id, round, join_id, indexed_probe ? 1 : 0,
			 hash_cost, nestloop_cost, cost_ratio_pct,
			 max_nestloop_cost_ratio_pct);
		if (max_nestloop_cost_ratio_pct > 0 &&
			cost_ratio_pct > (double) max_nestloop_cost_ratio_pct)
		{
			elog(LOG, "[nqo] run=" UINT64_FORMAT
				 " round %d: adaptive candidate join=%d rejected by "
				 "nest-loop cost guard",
				 run_id, round, join_id);
			bms_free(join_relids);
			bms_free(build_relids);
			return plan;
		}

		adaptive = nqo_make_adaptive_join(plan, candidate, build_plan,
											join_relids, level,
											threshold_rows, indexed_probe,
											run_id, round, join_id);
		(*wrapped)++;
		bms_free(join_relids);
		bms_free(build_relids);
		return (Plan *) adaptive;
	}

	bms_free(join_relids);
	bms_free(build_relids);
	return plan;
}

static int
nqo_count_hashjoins(Plan *plan)
{
	int			count = 0;

	if (plan == NULL)
		return 0;
	if (IsA(plan, HashJoin))
	{
		HashJoin   *hashjoin = (HashJoin *) plan;
		Plan	   *hash_node = innerPlan(plan);

		if (hashjoin->join.jointype == JOIN_INNER &&
			hash_node != NULL && IsA(hash_node, Hash) &&
			outerPlan(hash_node) != NULL &&
			!plan->parallel_aware &&
			plan->extParam == NULL && plan->initPlan == NIL)
			count++;
	}
	count += nqo_count_hashjoins(outerPlan(plan));
	count += nqo_count_hashjoins(innerPlan(plan));
	return count;
}

int
nqo_count_adaptive_hashjoins(PlannedStmt *plannedstmt)
{
	if (plannedstmt == NULL ||
		plannedstmt->commandType != CMD_SELECT ||
		plannedstmt->subplans != NIL ||
		plannedstmt->parallelModeNeeded)
		return 0;
	return nqo_count_hashjoins(plannedstmt->planTree);
}

static bool
nqo_merge_param_types(PlannedStmt *baseline, PlannedStmt *alternative)
{
	int			common;
	int			i;

	common = Min(list_length(baseline->paramExecTypes),
				 list_length(alternative->paramExecTypes));
	for (i = 0; i < common; i++)
	{
		if (list_nth_oid(baseline->paramExecTypes, i) !=
			list_nth_oid(alternative->paramExecTypes, i))
			return false;
	}
	for (i = common; i < list_length(alternative->paramExecTypes); i++)
		baseline->paramExecTypes =
			lappend_oid(baseline->paramExecTypes,
						list_nth_oid(alternative->paramExecTypes, i));
	return true;
}

int
nqo_wrap_adaptive_joins(PlannedStmt *baseline,
						   PlannedStmt *nestloop_alternative,
						   const char *level, int threshold_rows,
						   int max_nestloop_cost_ratio_pct,
						   uint64 run_id, int round)
{
	int			next_join_id = 1;
	int			wrapped = 0;

	if (baseline == NULL || nestloop_alternative == NULL ||
		baseline->commandType != CMD_SELECT ||
		nestloop_alternative->commandType != CMD_SELECT ||
		baseline->subplans != NIL || nestloop_alternative->subplans != NIL ||
		baseline->parallelModeNeeded ||
		nestloop_alternative->parallelModeNeeded ||
		threshold_rows < 1)
		return 0;

	if (!nqo_merge_param_types(baseline, nestloop_alternative))
		return 0;

	nqo_register_adaptive_join();
	baseline->planTree =
		nqo_wrap_plan_tree(baseline->planTree,
							  nestloop_alternative->planTree,
							  level, threshold_rows,
							  max_nestloop_cost_ratio_pct,
							  run_id, round,
							  &next_join_id, &wrapped);
	return wrapped;
}
