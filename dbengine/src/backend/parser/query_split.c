/*-------------------------------------------------------------------------
 *
 *
 *
 *
 * IDENTIFICATION
 *	  src/backend/parser/query_split.c
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"
#include "parser/query_split.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <netdb.h>
#include <sys/time.h>

#include "access/stratnum.h"
#include "executor/nodeNeurqoAdaptiveJoin.h"
#include "fe_utils/simple_list.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "commands/event_trigger.h"
#include "commands/portalcmds.h"
#include "utils/relmapper.h"
#include "commands/vacuum.h"
#include "catalog/pg_class.h"
#include "lib/stringinfo.h"
#include "parser/parse_func.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/ruleutils.h"
#include "utils/syscache.h"
#include "storage/fd.h"
#include "storage/buf_internals.h"
#include "parser/parse_relation.h"		/* addRTEPermissionInfo (PG16) */

#define NEWBETTER 1
#define OLDBETTER 2
#define NEURQO_MAX_LIP_FILTERS 10
#define NEURQO_MAX_LIP_PROBES 32
#define NEURQO_SEARCH_ABS_MAX_RELS 16
#define NEURQO_SEARCH_ABS_MAX_K 16
#define NEURQO_AJA_PLAN_MAX_NODES 96

//Create a local query
static Query* createQuery(const Query* querytree, CommandDest dest, List* rtable, Index* transfer_array, int length);
//change the RangeTblEntry relid to the new one
static void dochange(RangeTblEntry* rte, char* relname, Relation relation, Oid relid);
//Get the var will link to unlocal table
static List* findvarlist(List* joinlist, Index* transfer_array, int length);
//from postgres.c
extern void finish_xact_command();
//Get local rtable
static List* getRT_1(List* global_rtable, bool* graph, int length, int i, int j, Index* transfer_array);
static List* getRT_2(List* global_rtable, bool* graph, int length, int i, Index* transfer_array);
//Get local rtables' foreign keys
static List* grFK(List* rtable);
//is this subquery is the last ?
static int hasNext(bool* graph, int length);
//Is a restrict clause ?
static bool is_RC(Expr* expr);
//Transefer jointree to graph
static bool* List2Graph(bool* is_relationship, List* joinlist, List* FKlist, int length);
static int neurqo_remove_redundant_rr_equalities(
	Query* query, bool* relationship_flags, int length);
//Make a aggregation function as result
static List* removeAggref(List* targetList);
//give the new value to some var, prepare for the next subquery
static List* Prepare4Next(Query* global_query, Index* transfer_array, DR_intorel* receiver, PlannedStmt* plannedstmt,char* relname, List* FKlist);
static void Recon(const char* query_string, CommandTag commandTag, Node* pstmt, Query* ori_query, QueryCompletion* completionTag);
//remove redundant join
static void rRj(Query* querytree);
//Transfer fromlist to the local
static List* setfromlist(List* fromlist, Index* transfer_array, int length);
//Transfer global var to local
static List* setjoinlist(List* rclist, CommandDest dest, Index* transfer_array, int length);
//Make a local target list
static List* settargetlist(const List* global_rtable, List* local_rtable, CommandDest dest, List* varlist, List* targetlist, Index* transfer_array, int length);
//Remove used jointree
static List* simplifyjoinlist(List* list, CommandDest dest, Index* transfer_array, bool* graph, int length);
//Split Query by Foreign Key
static List* spq(char* query_string, CommandTag commandTag, Node* pstmt, Query* querytree, QueryCompletion* completionTag);
//from postgres.c
extern void start_xact_command();
static int tarfunc(Index* rels, PlannedStmt* new, PlannedStmt* old);
//Execute the local query
static List* QSExecutor(const char* query_string, CommandTag commandTag, Node* pstmt, PlannedStmt* plannedstmt, CommandDest dest, char* relname, QueryCompletion* completionTag, Query* querytree, Index* transfer_array, List* FKlist, MemoryContext oldcontext);
//Generate the current QSA candidates and select one for execution
static Query* QSSelectSubquery(Query* global_query, bool* graph,
							   Index* transfer_array, int length,
							   const char* query_string, int round,
							   double cumulative_cost_ms,
							   int max_split_rounds,
							   double* policy_ms,
							   char** selection_state_json_out);
static bool neurqo_split_candidate_valid(
	Query* query, int center_x, int center_y);
static Plan* find_node_with_nleaf_recursive(Plan* plan, int nleaf, int* leaf_has, int* depth);
static void walk_plantree(Plan* plan, Index* rel);
static PlannedStmt* neurqo_plan(Query* q, int cursorOptions, bool apply_lip);
static PlannedStmt* neurqo_plan_direct(Query* q, int cursorOptions,
									   bool apply_lip,
									   const char* hint_query_string,
									   bool log_hint);
static PlannedStmt* neurqo_plan_nestloop_candidate(
	Query* q, const char* hint_query_string);
static PlannedStmt* neurqo_plan_hashjoin_candidate(
	Query* q, const char* hint_query_string);
static char* neurqo_build_planner_hint(Query* q);
static char* neurqo_build_search_hint(Query* q,
									  PlannedStmt** selected_plan_out);
static char* neurqo_build_join_method_hint(Query* q, const char* method);
static char* neurqo_build_plan_hint(PlannedStmt* plannedstmt, Query* q);
static char* neurqo_build_leading_hint(PlannedStmt* plannedstmt, Query* q,
									  bool swap_hash_inputs);
static const char* neurqo_order_decision_name(int mode);
static bool neurqo_search_enabled(void);
static bool neurqo_aja_enabled(void);
static const char* neurqo_adaptive_aja_level(void);
static bool neurqo_lip_enabled(void);
static void neurqo_reset_execution_actions(void);
static bool neurqo_policy_high(Query* q, const char* query_string,
							   int round, int length, int remaining,
							   double cumulative_cost_ms,
							   int max_split_rounds,
							   bool* stop_now, double* policy_ms,
							   char** state_json_out);
static bool neurqo_policy_select(Query* q, const char* query_string,
								 int round, List* candidates,
								 double cumulative_cost_ms,
								 int max_split_rounds,
								 int* candidate_id,
								 double* policy_ms,
								 char** state_json_out);
static bool neurqo_policy_search(Query* q, const char* query_string,
								 int round, int length, int remaining,
								 double cumulative_cost_ms,
								 int max_split_rounds,
								 double* policy_ms,
								 char** state_json_out);
static bool neurqo_policy_low(Query* q,
							  int round, PlannedStmt* selected_plan,
							  double cumulative_cost_ms,
							  int max_split_rounds,
							  bool is_split_execution,
							  char** aja_hint_out,
							  double* policy_ms,
							  char** state_json_out);
static PlannedStmt* neurqo_plan_execution(Query* q, const char* query_string,
										  int round, int length, int remaining,
										  double cumulative_cost_ms,
										  int max_split_rounds,
										  bool is_split_execution,
										  double* policy_ms,
										  char** search_state_json_out,
										  char** low_state_json_out);
static double neurqo_now_ms(void);
static bool neurqo_apply_lip(Query* q, PlannedStmt* reference_plan,
							 double* lip_ms, int* lip_filters);
static char* neurqo_make_hint_query(const char* first_hint,
									const char* second_hint);
static char* neurqo_build_round_state(Query* q, const char* query_string,
									  const char* request_type,
									  int round, int length, int remaining,
									  double cumulative_cost_ms,
									  int max_split_rounds);
static void neurqo_append_relations_json(Query* q, StringInfo out);
static char* neurqo_build_selection_state(Query* q, const char* query_string,
										  int round, List* candidates,
										  double cumulative_cost_ms,
										  int max_split_rounds);
static char* neurqo_build_low_state(Query* q,
									int round, PlannedStmt* selected_plan,
									double cumulative_cost_ms,
									int max_split_rounds,
									bool is_split_execution);
static void neurqo_plan_summary(Plan* plan, int depth, int* nnodes,
								int* njoins, int* nscans, int* max_depth);
static void neurqo_append_aliases_json(Query* q, StringInfo out);
static void neurqo_append_plan_json(Plan* plan, Query* q, StringInfo out,
									int* nnodes);
static void neurqo_log_trajectory_event(const char* phase, int round,
										const char* state_json, bool stop_now,
										const char* selection_state_json,
										const char* search_state_json,
										const char* low_state_json,
										PlannedStmt* execution_plan,
										Query* execution_query,
										double policy_ms, double planning_ms,
										double execution_ms, double total_ms,
										const char* result);
static void neurqo_reset_execution_metrics(void);
static void neurqo_analyze_temp_relation(Oid relid, RangeVar* relation);
static int64 neurqo_total_relation_size(Oid relid);
static Index neurqo_source_varno(const Var* var, int length);
static AttrNumber neurqo_source_attno(const Var* var);

bool* is_relationship;
int query_splitting_algorithm = None;
int order_decision = only_cost;
bool neurqo_enabled = false;		/* backing var for the `neurqo` GUC */
char* neurqo_server_url = NULL;
char* neurqo_trajectory_log_path = NULL;
int neurqo_server_timeout_ms = 2000;
int neurqo_max_rounds = 64;
int neurqo_search_topk = 5;
int neurqo_search_max_rels = 12;
bool neurqo_search_exact_cardinality = false;
int neurqo_aja_conservative_rows = 362443;
int neurqo_aja_aggressive_rows = 3624434;
int neurqo_aja_max_nestloop_cost_ratio_pct = 150;
int neurqo_aja_aggressive_max_nestloop_cost_ratio_pct = 125;
int neurqo_lip_max_build_relation_rows = 500000;
int neurqo_lip_selective_plan_rows = 10000;
int neurqo_lip_max_build_selectivity_pct = 10;
int neurqo_lip_min_probe_ratio = 2;
int neurqo_lip_max_filters = 4;
static char neurqo_current_search_strategy[64] = "";
static char neurqo_current_execution_action[64] = "";
static char neurqo_current_lip_action[64] = "";
static char neurqo_current_high_action[16] = "";
static char neurqo_current_selection_strategy[64] = "";
static int neurqo_current_search_k = 0;
static int neurqo_current_candidate_id = -1;
//the number of subquery
static int queryId = 0;
static uint64 neurqo_run_seq = 0;
static uint64 neurqo_current_run_id = 0;
static double neurqo_last_executor_ms = 0.0;
static double neurqo_last_analyze_ms = 0.0;
static double neurqo_last_residual_rewrite_ms = 0.0;
static double neurqo_last_search_ms = 0.0;
static double neurqo_last_search_candidate_cost = 0.0;
static int neurqo_last_search_candidates = 0;
static int neurqo_last_search_planner_calls = 0;
static bool neurqo_last_search_applied = false;
static double neurqo_last_lip_build_ms = 0.0;
static int neurqo_last_lip_filters = 0;
static int neurqo_last_adaptive_joins = 0;
static int neurqo_last_adaptive_threshold = 0;
static uint64 neurqo_last_materialized_rows = 0;
static int64 neurqo_last_materialized_bytes = 0;
//where to send the result, to the client end or temporary table
CommandDest mydest;
Index* transfer_array = NULL;

typedef struct NeurqoPolicyAction
{
	char action[64];
	bool stop;
	bool has_order_decision;
	int order_decision;
	bool has_candidate_id;
	int candidate_id;
	bool has_selection_strategy;
	char selection_strategy[64];
	bool has_search_strategy;
	char search_strategy[64];
	bool has_search_k;
	int search_k;
	bool has_execution_action;
	char execution_action[64];
	bool has_lip_action;
	char lip_action[64];
	bool has_aja_hint;
	char aja_hint[1024];
	bool has_join_method;
	char join_method[64];
	char note[256];
} NeurqoPolicyAction;

typedef struct NeurqoSplitCandidate
{
	int			candidate_id;
	int			x;
	int			y;
	Query	   *query;
	PlannedStmt *estimate_plan;
} NeurqoSplitCandidate;

typedef struct NeurqoLipFilter
{
	int			filter_id;
	bool		is_build;
	Var		   *build_var;
	Var		   *probe_var;
} NeurqoLipFilter;

typedef struct NeurqoLipRelStats
{
	bool		has_plan;
	double		plan_rows;
	double		plan_cost;
	double		relation_rows;
} NeurqoLipRelStats;

typedef struct NeurqoVarnoRemapContext
{
	Index		from_varno;
	Index		to_varno;
} NeurqoVarnoRemapContext;

typedef struct NeurqoSearchRel
{
	Index		rtindex;
	char	   *alias;
} NeurqoSearchRel;

typedef struct NeurqoSearchEntry
{
	double		cout;
	char	   *leading;
} NeurqoSearchEntry;

typedef struct NeurqoSearchCell
{
	int			nentries;
	NeurqoSearchEntry entries[NEURQO_SEARCH_ABS_MAX_K];
	bool		card_valid;
	double		card_rows;
} NeurqoSearchCell;

static double
neurqo_now_ms(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	return (double) tv.tv_sec * 1000.0 + (double) tv.tv_usec / 1000.0;
}

static void
neurqo_append_json_string(StringInfo dst, const char* value)
{
	const unsigned char* p;

	if (value == NULL)
	{
		appendStringInfoString(dst, "null");
		return;
	}
	appendStringInfoChar(dst, '"');
	for (p = (const unsigned char*)value; *p; p++)
	{
		switch (*p)
		{
			case '"':
				appendStringInfoString(dst, "\\\"");
				break;
			case '\\':
				appendStringInfoString(dst, "\\\\");
				break;
			case '\b':
				appendStringInfoString(dst, "\\b");
				break;
			case '\f':
				appendStringInfoString(dst, "\\f");
				break;
			case '\n':
				appendStringInfoString(dst, "\\n");
				break;
			case '\r':
				appendStringInfoString(dst, "\\r");
				break;
			case '\t':
				appendStringInfoString(dst, "\\t");
				break;
			default:
				if (*p < 0x20)
					appendStringInfo(dst, "\\u%04x", *p);
				else
					appendStringInfoChar(dst, *p);
				break;
		}
	}
	appendStringInfoChar(dst, '"');
}

static char*
neurqo_build_round_state(Query* q, const char* query_string,
						 const char* request_type,
						 int round, int length, int remaining,
						 double cumulative_cost_ms,
						 int max_split_rounds)
{
	StringInfoData state;
	ListCell* lc;
	char* current_sql;
	int base_rels = 0;

	foreach(lc, q->rtable)
	{
		RangeTblEntry* rte = (RangeTblEntry*)lfirst(lc);

		if (rte->rtekind == RTE_RELATION)
			base_rels++;
	}
	if (base_rels == 0)
		base_rels = length;

	initStringInfo(&state);
	current_sql = pg_get_querydef(q, false);
	appendStringInfo(&state,
					 "{\"pid\":%d,\"run_id\":" UINT64_FORMAT
					 ",\"request_type\":\"%s\",\"round\":%d,"
					 "\"base_rels\":%d,\"remaining_splits\":%d,"
					 "\"cumulative_cost_ms\":%.3f,\"max_split_rounds\":%d,"
					 "\"search_max_rels\":%d,"
					 "\"algorithm\":%d,\"order_decision\":\"%s\","
					 "\"sql\":",
					 MyProcPid, neurqo_current_run_id, request_type, round,
					 base_rels, remaining, cumulative_cost_ms, max_split_rounds,
					 neurqo_search_max_rels,
					 query_splitting_algorithm,
					 neurqo_order_decision_name(order_decision));
	neurqo_append_json_string(&state,
							  current_sql != NULL ? current_sql : query_string);
	appendStringInfoString(&state, ",\"relations\":");
	neurqo_append_relations_json(q, &state);
	appendStringInfoString(&state, "}");
	if (current_sql != NULL)
		pfree(current_sql);
	return state.data;
}

static void
neurqo_append_relations_json(Query* q, StringInfo out)
{
	ListCell* lc;
	bool first = true;

	appendStringInfoChar(out, '[');
	foreach(lc, q->rtable)
	{
		RangeTblEntry* rte = (RangeTblEntry*)lfirst(lc);
		const char* alias;
		char* relname = NULL;
		HeapTuple tuple;
		Form_pg_class classform = NULL;
		double estimated_rows = 0.0;
		int relpages = 0;
		bool is_temporary = false;
		int64 relation_bytes = 0;

		if (rte->rtekind != RTE_RELATION)
			continue;
		alias = rte->eref ? rte->eref->aliasname : NULL;
		relname = get_rel_name(rte->relid);
		tuple = SearchSysCache1(RELOID, ObjectIdGetDatum(rte->relid));
		if (HeapTupleIsValid(tuple))
		{
			classform = (Form_pg_class)GETSTRUCT(tuple);
			estimated_rows = classform->reltuples;
			relpages = classform->relpages;
			is_temporary =
				classform->relpersistence == RELPERSISTENCE_TEMP;
			if (is_temporary)
				relation_bytes = neurqo_total_relation_size(rte->relid);
		}
		if (!first)
			appendStringInfoChar(out, ',');
		first = false;
		appendStringInfoString(out, "{\"alias\":");
		neurqo_append_json_string(out, alias);
		appendStringInfoString(out, ",\"relname\":");
		neurqo_append_json_string(out, relname ? relname : alias);
		appendStringInfo(out,
						 ",\"relid\":%u,\"estimated_rows\":%.0f,"
						 "\"pages\":%d,\"is_temporary\":%s,"
						 "\"bytes\":" INT64_FORMAT "}",
						 rte->relid, estimated_rows, relpages,
						 is_temporary ? "true" : "false",
						 relation_bytes);
		if (HeapTupleIsValid(tuple))
			ReleaseSysCache(tuple);
		if (relname)
			pfree(relname);
	}
	appendStringInfoChar(out, ']');
}

static char*
neurqo_build_selection_state(Query* q, const char* query_string,
							 int round, List* candidates,
							 double cumulative_cost_ms,
							 int max_split_rounds)
{
	StringInfoData state;
	ListCell* lc;
	char* current_sql = pg_get_querydef(q, false);
	bool first = true;

	initStringInfo(&state);
	appendStringInfo(&state,
					 "{\"pid\":%d,\"run_id\":" UINT64_FORMAT
					 ",\"request_type\":\"select\",\"round\":%d,"
					 "\"candidate_count\":%d,\"cumulative_cost_ms\":%.3f,"
					 "\"max_split_rounds\":%d,\"sql\":",
					 MyProcPid, neurqo_current_run_id, round,
					 list_length(candidates), cumulative_cost_ms,
					 max_split_rounds);
	neurqo_append_json_string(&state,
							  current_sql != NULL ? current_sql : query_string);
	appendStringInfoString(&state, ",\"original_sql\":");
	neurqo_append_json_string(&state, query_string);
	appendStringInfoString(&state, ",\"relations\":");
	neurqo_append_relations_json(q, &state);
	appendStringInfoString(&state, ",\"candidates\":[");
	foreach(lc, candidates)
	{
		NeurqoSplitCandidate* candidate =
			(NeurqoSplitCandidate*)lfirst(lc);
		Plan* plan = candidate->estimate_plan != NULL ?
			candidate->estimate_plan->planTree : NULL;
		char* candidate_sql = pg_get_querydef(candidate->query, false);
		double rows = plan != NULL ? Max(plan->plan_rows, 1.0) : 1.0;
		double cost = plan != NULL ? plan->total_cost : DBL_MAX;
		double phi4_score =
			cost > DBL_MAX / rows ? DBL_MAX : cost * rows;

		if (!first)
			appendStringInfoChar(&state, ',');
		first = false;
		appendStringInfo(&state,
						 "{\"candidate_id\":%d,\"center_index\":%d,"
						 "\"plan_total_cost\":%.6g,\"plan_rows\":%.6g,"
						 "\"phi4_score\":%.6g,\"sql\":",
						 candidate->candidate_id, candidate->x,
						 cost, rows, phi4_score);
		neurqo_append_json_string(&state, candidate_sql);
		appendStringInfoString(&state, ",\"aliases\":");
		neurqo_append_aliases_json(candidate->query, &state);
		appendStringInfoChar(&state, '}');
		if (candidate_sql != NULL)
			pfree(candidate_sql);
	}
	appendStringInfoString(&state, "]}");
	if (current_sql != NULL)
		pfree(current_sql);
	return state.data;
}

static void
neurqo_log_trajectory_event(const char* phase, int round,
							const char* state_json, bool stop_now,
							const char* selection_state_json,
							const char* search_state_json,
							const char* low_state_json,
							PlannedStmt* execution_plan,
							Query* execution_query,
							double policy_ms, double planning_ms,
							double execution_ms, double total_ms,
							const char* result)
{
	FILE* fp;
	StringInfoData line;
	NeurqoAdaptiveJoinStats aja_stats = neurqo_get_adaptive_join_stats();

	if (neurqo_trajectory_log_path == NULL ||
		neurqo_trajectory_log_path[0] == '\0')
		return;

	fp = AllocateFile(neurqo_trajectory_log_path, "a");
	if (fp == NULL)
	{
		elog(WARNING, "[neurqo] run=" UINT64_FORMAT
			 " could not append trajectory log %s: %m",
			 neurqo_current_run_id, neurqo_trajectory_log_path);
		return;
	}

	initStringInfo(&line);
	appendStringInfo(&line,
					 "{\"ts_ms\":%.3f,\"pid\":%d,\"run_id\":"
					 UINT64_FORMAT ",\"phase\":",
					 neurqo_now_ms(), MyProcPid, neurqo_current_run_id);
	neurqo_append_json_string(&line, phase);
	appendStringInfo(&line, ",\"round\":%d,\"stop\":%s,\"state\":",
					 round, stop_now ? "true" : "false");
	if (state_json != NULL && state_json[0] != '\0')
		appendStringInfoString(&line, state_json);
	else
		appendStringInfoString(&line, "null");
	appendStringInfoString(&line, ",\"decision_states\":{\"high\":");
	if (state_json != NULL && state_json[0] != '\0' &&
		neurqo_current_high_action[0] != '\0')
		appendStringInfoString(&line, state_json);
	else
		appendStringInfoString(&line, "null");
	appendStringInfoString(&line, ",\"select\":");
	if (selection_state_json != NULL && selection_state_json[0] != '\0')
		appendStringInfoString(&line, selection_state_json);
	else
		appendStringInfoString(&line, "null");
	appendStringInfoString(&line, ",\"search\":");
	if (search_state_json != NULL && search_state_json[0] != '\0')
		appendStringInfoString(&line, search_state_json);
	else
		appendStringInfoString(&line, "null");
	appendStringInfoString(&line, ",\"low\":");
	if (low_state_json != NULL && low_state_json[0] != '\0')
		appendStringInfoString(&line, low_state_json);
	else
		appendStringInfoString(&line, "null");
	appendStringInfoChar(&line, '}');
	appendStringInfoString(&line, ",\"execution_plan\":");
	if (execution_plan != NULL && execution_plan->planTree != NULL &&
		execution_query != NULL)
	{
		int nnodes = 0;

		neurqo_append_plan_json(execution_plan->planTree, execution_query,
								&line, &nnodes);
	}
	else
		appendStringInfoString(&line, "null");
	appendStringInfoString(&line, ",\"action\":{");
	appendStringInfoString(&line, "\"high_action\":");
	neurqo_append_json_string(&line,
							  neurqo_current_high_action[0] != '\0' ?
							  neurqo_current_high_action : NULL);
	appendStringInfoString(&line, ",\"order_decision\":");
	neurqo_append_json_string(&line, neurqo_order_decision_name(order_decision));
	appendStringInfoString(&line, ",\"candidate_id\":");
	if (selection_state_json == NULL)
		appendStringInfoString(&line, "null");
	else
		appendStringInfo(&line, "%d", neurqo_current_candidate_id);
	appendStringInfoString(&line, ",\"selection_strategy\":");
	if (selection_state_json == NULL)
		neurqo_append_json_string(&line, NULL);
	else
		neurqo_append_json_string(
			&line,
			neurqo_current_selection_strategy[0] != '\0' ?
			neurqo_current_selection_strategy : "phi4");
	appendStringInfoString(&line, ",\"search_strategy\":");
	if (search_state_json == NULL)
		neurqo_append_json_string(&line, NULL);
	else
		neurqo_append_json_string(&line,
								  neurqo_search_enabled() ?
								  neurqo_current_search_strategy : "default");
	if (search_state_json == NULL)
		appendStringInfoString(&line, ",\"search_k\":null");
	else
		appendStringInfo(&line, ",\"search_k\":%d",
						 neurqo_current_search_k > 0 ?
						 neurqo_current_search_k : 0);
	appendStringInfoString(&line, ",\"execution_action\":");
	if (low_state_json == NULL)
		neurqo_append_json_string(&line, NULL);
	else
		neurqo_append_json_string(&line,
								  neurqo_aja_enabled() ?
								  neurqo_current_execution_action : "none");
	appendStringInfoString(&line, ",\"lip_action\":");
	if (low_state_json == NULL)
		neurqo_append_json_string(&line, NULL);
	else
		neurqo_append_json_string(&line,
								  neurqo_lip_enabled() ?
								  neurqo_current_lip_action : "none");
	appendStringInfo(&line, ",\"lip_filters\":%d",
					 neurqo_last_lip_filters);
	appendStringInfo(&line, ",\"aja_threshold_rows\":%d",
					 neurqo_last_adaptive_threshold);
	appendStringInfo(&line,
					 ",\"search_applied\":%s,\"search_candidates\":%d,"
					 "\"search_planner_calls\":%d,"
					 "\"search_cardinality_mode\":\"%s\","
					 "\"search_candidate_cost\":%.3f",
					 neurqo_last_search_applied ? "true" : "false",
					 neurqo_last_search_candidates,
					 neurqo_last_search_planner_calls,
					 neurqo_search_exact_cardinality ?
					 "exact" : "pairwise",
					 neurqo_last_search_candidate_cost);
	appendStringInfoString(&line, "},\"timing_ms\":{");
	appendStringInfo(&line,
					 "\"policy\":%.3f,\"search\":%.3f,\"planning\":%.3f,"
					 "\"lip_build\":%.3f,\"aja_build\":%.3f,"
					 "\"execution\":%.3f,\"executor\":%.3f,"
					 "\"analyze\":%.3f,\"residual_rewrite\":%.3f,"
					 "\"total\":%.3f}",
					 policy_ms, neurqo_last_search_ms, planning_ms,
					 neurqo_last_lip_build_ms,
					 aja_stats.build_ms, execution_ms,
					 neurqo_last_executor_ms, neurqo_last_analyze_ms,
					 neurqo_last_residual_rewrite_ms, total_ms);
	appendStringInfo(&line,
					 ",\"aja\":{\"planned\":%d,\"decided\":%d,"
					 "\"nestloop\":%d,\"hashjoin\":%d,"
					 "\"actual_build_rows\":" UINT64_FORMAT "}",
					 neurqo_last_adaptive_joins,
					 aja_stats.joins_decided,
					 aja_stats.nestloop_selected,
					 aja_stats.hashjoin_selected,
					 aja_stats.actual_build_rows);
	appendStringInfo(&line,
					 ",\"materialized\":{\"rows\":" UINT64_FORMAT
					 ",\"bytes\":" INT64_FORMAT "}",
					 neurqo_last_materialized_rows,
					 neurqo_last_materialized_bytes);
	appendStringInfoString(&line, ",\"result\":");
	neurqo_append_json_string(&line, result);
	appendStringInfoChar(&line, '}');

	fputs(line.data, fp);
	fputc('\n', fp);
	FreeFile(fp);
	pfree(line.data);
}

static void
neurqo_reset_execution_metrics(void)
{
	neurqo_last_executor_ms = 0.0;
	neurqo_last_analyze_ms = 0.0;
	neurqo_last_residual_rewrite_ms = 0.0;
	neurqo_last_materialized_rows = 0;
	neurqo_last_materialized_bytes = 0;
	neurqo_reset_adaptive_join_stats();
}

static void
neurqo_analyze_temp_relation(Oid relid, RangeVar* relation)
{
	VacuumParams params;

	memset(&params, 0, sizeof(params));
	params.options = VACOPT_ANALYZE;
	params.freeze_min_age = -1;
	params.freeze_table_age = -1;
	params.multixact_freeze_min_age = -1;
	params.multixact_freeze_table_age = -1;
	params.log_min_duration = -1;
	params.index_cleanup = VACOPTVALUE_UNSPECIFIED;
	params.truncate = VACOPTVALUE_UNSPECIFIED;
	params.nworkers = -1;
	analyze_rel(relid, relation, &params, NIL, true, NULL);
}

static int64
neurqo_total_relation_size(Oid relid)
{
	Oid argtypes[1] = {REGCLASSOID};
	Oid funcid;

	funcid = LookupFuncName(
		list_make1(makeString("pg_total_relation_size")),
		1, argtypes, false);
	return DatumGetInt64(OidFunctionCall1(funcid, ObjectIdGetDatum(relid)));
}

static const char*
neurqo_order_decision_name(int mode)
{
	switch (mode)
	{
		case only_cost:
			return "only_cost";
		case only_row:
			return "only_row";
		case hybrid_row:
			return "hybrid_row";
		case hybrid_sqrt:
			return "hybrid_sqrt";
		case hybrid_log:
			return "hybrid_log";
		case global_view:
			return "global_view";
		default:
			return "unknown";
	}
}

static bool
neurqo_search_enabled(void)
{
	return neurqo_current_search_strategy[0] != '\0' &&
		strcmp(neurqo_current_search_strategy, "default") != 0 &&
		strcmp(neurqo_current_search_strategy, "none") != 0;
}

static bool
neurqo_aja_enabled(void)
{
	return neurqo_current_execution_action[0] != '\0' &&
		(neurqo_adaptive_aja_level() != NULL ||
		 strcmp(neurqo_current_execution_action, "hashjoin") == 0 ||
		 strcmp(neurqo_current_execution_action, "nestloop") == 0 ||
		 strcmp(neurqo_current_execution_action, "mergejoin") == 0);
}

static const char*
neurqo_adaptive_aja_level(void)
{
	if (strcmp(neurqo_current_execution_action, "conservative") == 0)
		return "conservative";
	if (strcmp(neurqo_current_execution_action, "aggressive") == 0 ||
		strcmp(neurqo_current_execution_action, "aja") == 0)
		return "aggressive";
	return NULL;
}

static bool
neurqo_lip_enabled(void)
{
	return neurqo_current_lip_action[0] != '\0' &&
		strcmp(neurqo_current_lip_action, "default") != 0 &&
		strcmp(neurqo_current_lip_action, "none") != 0 &&
		strcmp(neurqo_current_lip_action, "off") != 0;
}

static bool
neurqo_parse_order_decision(const char* value, int* mode)
{
	if (strcmp(value, "only_cost") == 0 || strcmp(value, "0") == 0)
		*mode = only_cost;
	else if (strcmp(value, "only_row") == 0 || strcmp(value, "1") == 0)
		*mode = only_row;
	else if (strcmp(value, "hybrid_row") == 0 || strcmp(value, "2") == 0)
		*mode = hybrid_row;
	else if (strcmp(value, "hybrid_sqrt") == 0 || strcmp(value, "3") == 0)
		*mode = hybrid_sqrt;
	else if (strcmp(value, "hybrid_log") == 0 || strcmp(value, "4") == 0)
		*mode = hybrid_log;
	else if (strcmp(value, "global_view") == 0 || strcmp(value, "5") == 0)
		*mode = global_view;
	else
		return false;
	return true;
}

static bool
neurqo_parse_http_url(const char* url, char* host, size_t hostlen,
					  char* port, size_t portlen, char* path, size_t pathlen)
{
	const char* p = url;
	const char* slash;
	const char* colon;
	const char* hostend;
	size_t hl;

	if (strncmp(p, "http://", 7) == 0)
		p += 7;

	slash = strchr(p, '/');
	colon = strchr(p, ':');
	if (colon && (!slash || colon < slash))
	{
		const char* pe = slash ? slash : p + strlen(p);
		size_t pl = (size_t)(pe - (colon + 1));

		hostend = colon;
		if (pl == 0 || pl >= portlen)
			return false;
		memcpy(port, colon + 1, pl);
		port[pl] = '\0';
	}
	else
	{
		hostend = slash ? slash : p + strlen(p);
		snprintf(port, portlen, "80");
	}

	hl = (size_t)(hostend - p);
	if (hl == 0 || hl >= hostlen)
		return false;
	memcpy(host, p, hl);
	host[hl] = '\0';

	if (slash)
	{
		if (strlen(slash) >= pathlen)
			return false;
		strcpy(path, slash);
	}
	else
		snprintf(path, pathlen, "/");

	return true;
}

static bool
neurqo_http_post(const char* url, const char* body, StringInfo resp,
				 char* errbuf, size_t errlen)
{
	char host[256];
	char port[16];
	char path[256];
	struct addrinfo hints;
	struct addrinfo* res = NULL;
	struct addrinfo* rp;
	int gai;
	int fd = -1;
	StringInfoData req;
	StringInfoData raw;
	char buf[4096];
	size_t total;
	char* sep;

	if (url == NULL || url[0] == '\0')
	{
		snprintf(errbuf, errlen, "neurqo.server_url is empty");
		return false;
	}
	if (!neurqo_parse_http_url(url, host, sizeof(host), port, sizeof(port),
							   path, sizeof(path)))
	{
		snprintf(errbuf, errlen, "bad server url: %s", url);
		return false;
	}

	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	gai = getaddrinfo(host, port, &hints, &res);
	if (gai != 0)
	{
		snprintf(errbuf, errlen, "getaddrinfo(%s:%s): %s",
				 host, port, gai_strerror(gai));
		return false;
	}

	for (rp = res; rp != NULL; rp = rp->ai_next)
	{
		struct timeval tv;

		fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
		if (fd < 0)
			continue;
		tv.tv_sec = neurqo_server_timeout_ms / 1000;
		tv.tv_usec = (neurqo_server_timeout_ms % 1000) * 1000;
		setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
		setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
		if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0)
			break;
		close(fd);
		fd = -1;
	}
	freeaddrinfo(res);

	if (fd < 0)
	{
		snprintf(errbuf, errlen, "connect %s:%s failed: %s",
				 host, port, strerror(errno));
		return false;
	}

	initStringInfo(&req);
	appendStringInfo(&req,
					 "POST %s HTTP/1.1\r\n"
					 "Host: %s:%s\r\n"
					 "Content-Type: application/json\r\n"
					 "Content-Length: %d\r\n"
					 "Connection: close\r\n"
					 "\r\n",
					 path, host, port, (int)strlen(body));
	appendStringInfoString(&req, body);

	total = 0;
	while (total < (size_t)req.len)
	{
		ssize_t w = send(fd, req.data + total, req.len - total, 0);

		if (w <= 0)
		{
			snprintf(errbuf, errlen, "send failed: %s", strerror(errno));
			close(fd);
			pfree(req.data);
			return false;
		}
		total += (size_t)w;
	}
	pfree(req.data);

	initStringInfo(&raw);
	for (;;)
	{
		ssize_t r = recv(fd, buf, sizeof(buf), 0);

		if (r > 0)
			appendBinaryStringInfo(&raw, buf, (int)r);
		else if (r == 0)
			break;
		else
		{
			if (raw.len == 0)
			{
				snprintf(errbuf, errlen, "recv failed: %s", strerror(errno));
				close(fd);
				pfree(raw.data);
				return false;
			}
			break;
		}
	}
	close(fd);

	sep = strstr(raw.data, "\r\n\r\n");
	if (sep == NULL)
	{
		snprintf(errbuf, errlen, "malformed http response");
		pfree(raw.data);
		return false;
	}
	appendStringInfoString(resp, sep + 4);
	pfree(raw.data);
	return true;
}

static void
neurqo_parse_policy_action(const char* body, NeurqoPolicyAction* act)
{
	const char* p = body;

	memset(act, 0, sizeof(*act));
	snprintf(act->action, sizeof(act->action), "none");
	act->stop = false;

	while (*p)
	{
		const char* eol = strchr(p, '\n');
		size_t linelen = eol ? (size_t)(eol - p) : strlen(p);
		char line[2048];
		size_t cl = linelen < sizeof(line) - 1 ? linelen : sizeof(line) - 1;
		char* eq;

		memcpy(line, p, cl);
		line[cl] = '\0';
		if (cl > 0 && line[cl - 1] == '\r')
			line[cl - 1] = '\0';

		eq = strchr(line, '=');
		if (eq != NULL)
		{
			char* key = line;
			char* val = eq + 1;

			*eq = '\0';
			if (strcmp(key, "action") == 0)
				snprintf(act->action, sizeof(act->action), "%s", val);
			else if (strcmp(key, "stop") == 0)
				act->stop = (atoi(val) != 0);
			else if (strcmp(key, "note") == 0)
				snprintf(act->note, sizeof(act->note), "%s", val);
			else if (strcmp(key, "order_decision") == 0)
			{
				int mode;

				if (neurqo_parse_order_decision(val, &mode))
				{
					act->has_order_decision = true;
					act->order_decision = mode;
				}
				else
					elog(WARNING, "[neurqo] AI server returned unknown order_decision=%s", val);
			}
			else if (strcmp(key, "candidate_id") == 0)
			{
				act->has_candidate_id = true;
				act->candidate_id = atoi(val);
			}
			else if (strcmp(key, "selection_strategy") == 0)
			{
				act->has_selection_strategy = true;
				snprintf(act->selection_strategy,
						 sizeof(act->selection_strategy), "%s", val);
			}
			else if (strcmp(key, "search_strategy") == 0)
			{
				act->has_search_strategy = true;
				snprintf(act->search_strategy, sizeof(act->search_strategy), "%s", val);
			}
			else if (strcmp(key, "search_k") == 0)
			{
				act->has_search_k = true;
				act->search_k = atoi(val);
			}
			else if (strcmp(key, "execution_action") == 0)
			{
				act->has_execution_action = true;
				snprintf(act->execution_action, sizeof(act->execution_action), "%s", val);
			}
			else if (strcmp(key, "lip_action") == 0)
			{
				act->has_lip_action = true;
				snprintf(act->lip_action, sizeof(act->lip_action), "%s", val);
			}
			else if (strcmp(key, "aja_hint") == 0 || strcmp(key, "hint") == 0)
			{
				act->has_aja_hint = true;
				snprintf(act->aja_hint, sizeof(act->aja_hint), "%s", val);
			}
			else if (strcmp(key, "join_method") == 0 ||
					 strcmp(key, "join_method_hint") == 0)
			{
				act->has_join_method = true;
				snprintf(act->join_method, sizeof(act->join_method), "%s", val);
			}
		}

		if (!eol)
			break;
		p = eol + 1;
	}
}

static void
neurqo_reset_execution_actions(void)
{
	neurqo_current_search_strategy[0] = '\0';
	neurqo_current_execution_action[0] = '\0';
	neurqo_current_lip_action[0] = '\0';
	neurqo_current_selection_strategy[0] = '\0';
	neurqo_current_search_k = 0;
	neurqo_current_candidate_id = -1;
}

static bool
neurqo_request_policy_action(const char* request_type, int round,
							 const char* state_json,
							 NeurqoPolicyAction* act,
							 double* policy_ms)
{
	StringInfoData resp;
	char errbuf[256];
	double t0;
	bool ok;

	*policy_ms = 0.0;
	initStringInfo(&resp);
	t0 = neurqo_now_ms();
	ok = neurqo_http_post(neurqo_server_url, state_json, &resp,
						  errbuf, sizeof(errbuf));
	*policy_ms = neurqo_now_ms() - t0;

	if (!ok)
	{
		elog(WARNING, "[neurqo] run=" UINT64_FORMAT
			 " round %d: %s policy call failed (%s)",
			 neurqo_current_run_id, round, request_type, errbuf);
		pfree(resp.data);
		return false;
	}

	neurqo_parse_policy_action(resp.data, act);
	elog(DEBUG1, "[neurqo] run=" UINT64_FORMAT
		 " round %d: %s policy response=%s policy_ms=%.2f",
		 neurqo_current_run_id, round, request_type, resp.data, *policy_ms);
	pfree(resp.data);
	return true;
}

static bool
neurqo_policy_high(Query* q, const char* query_string,
				   int round, int length, int remaining,
				   double cumulative_cost_ms, int max_split_rounds,
				   bool* stop_now, double* policy_ms,
				   char** state_json_out)
{
	NeurqoPolicyAction act;
	char* state_json;
	bool ok;

	neurqo_reset_execution_actions();
	neurqo_current_high_action[0] = '\0';
	*stop_now = false;
	if (state_json_out != NULL)
		*state_json_out = NULL;
	state_json = neurqo_build_round_state(q, query_string, "high",
										 round, length, remaining,
										 cumulative_cost_ms,
										 max_split_rounds);
	ok = neurqo_request_policy_action("high", round, state_json, &act,
									 policy_ms);
	if (!ok)
	{
		if (state_json_out != NULL)
			*state_json_out = state_json;
		else
			pfree(state_json);
		return false;
	}

	if (act.has_order_decision)
		order_decision = act.order_decision;
	if (remaining > 0 && strcmp(act.action, "split") == 0 && !act.stop)
	{
		snprintf(neurqo_current_high_action,
				 sizeof(neurqo_current_high_action), "split");
		*stop_now = false;
	}
	else
	{
		snprintf(neurqo_current_high_action,
				 sizeof(neurqo_current_high_action), "stop");
		*stop_now = true;
	}

	elog(LOG, "[neurqo] run=" UINT64_FORMAT
		" round %d: high action=%s stop=%d order_decision=%s note=\"%s\" policy_ms=%.2f",
		neurqo_current_run_id, round, neurqo_current_high_action,
		*stop_now ? 1 : 0,
		neurqo_order_decision_name(order_decision), act.note, *policy_ms);

	if (state_json_out != NULL)
		*state_json_out = state_json;
	else
		pfree(state_json);
	return true;
}

static bool
neurqo_policy_select(Query* q, const char* query_string,
					 int round, List* candidates,
					 double cumulative_cost_ms, int max_split_rounds,
					 int* candidate_id, double* policy_ms,
					 char** state_json_out)
{
	NeurqoPolicyAction act;
	char* state_json;
	bool ok;
	int ncandidates = list_length(candidates);

	*candidate_id = -1;
	neurqo_current_candidate_id = -1;
	neurqo_current_selection_strategy[0] = '\0';
	if (state_json_out != NULL)
		*state_json_out = NULL;
	state_json = neurqo_build_selection_state(
		q, query_string, round, candidates, cumulative_cost_ms,
		max_split_rounds);
	ok = neurqo_request_policy_action("select", round, state_json, &act,
									 policy_ms);
	if (!ok)
	{
		if (state_json_out != NULL)
			*state_json_out = state_json;
		else
			pfree(state_json);
		return false;
	}

	if (!act.has_candidate_id ||
		act.candidate_id < 0 || act.candidate_id >= ncandidates)
	{
		elog(WARNING, "[neurqo] run=" UINT64_FORMAT
			 " round %d: select returned invalid candidate_id=%d for %d candidates",
			 neurqo_current_run_id, round,
			 act.has_candidate_id ? act.candidate_id : -1, ncandidates);
		if (state_json_out != NULL)
			*state_json_out = state_json;
		else
			pfree(state_json);
		return false;
	}

	*candidate_id = act.candidate_id;
	neurqo_current_candidate_id = act.candidate_id;
	snprintf(neurqo_current_selection_strategy,
			 sizeof(neurqo_current_selection_strategy), "%s",
			 act.has_selection_strategy ?
			 act.selection_strategy : "model");
	elog(LOG, "[neurqo] run=" UINT64_FORMAT
		 " round %d: select candidate_id=%d/%d strategy=%s note=\"%s\" policy_ms=%.2f",
		 neurqo_current_run_id, round, *candidate_id, ncandidates,
		 neurqo_current_selection_strategy, act.note, *policy_ms);

	if (state_json_out != NULL)
		*state_json_out = state_json;
	else
		pfree(state_json);
	return true;
}

static bool
neurqo_policy_search(Query* q, const char* query_string,
					 int round, int length, int remaining,
					 double cumulative_cost_ms, int max_split_rounds,
					 double* policy_ms, char** state_json_out)
{
	NeurqoPolicyAction act;
	char* state_json;
	bool ok;

	neurqo_current_search_strategy[0] = '\0';
	neurqo_current_search_k = 0;
	if (state_json_out != NULL)
		*state_json_out = NULL;
	state_json = neurqo_build_round_state(q, query_string, "search",
										 round, length, remaining,
										 cumulative_cost_ms,
										 max_split_rounds);
	ok = neurqo_request_policy_action("search", round, state_json, &act,
									 policy_ms);
	if (!ok)
	{
		pfree(state_json);
		return false;
	}

	if (act.has_search_strategy)
		snprintf(neurqo_current_search_strategy,
				 sizeof(neurqo_current_search_strategy),
				 "%s", act.search_strategy);
	if (act.has_search_k && act.search_k > 0)
		neurqo_current_search_k = act.search_k;

	elog(LOG, "[neurqo] run=" UINT64_FORMAT
		 " round %d: search strategy=%s k=%d note=\"%s\" policy_ms=%.2f",
		 neurqo_current_run_id, round,
		 neurqo_search_enabled() ? neurqo_current_search_strategy : "default",
		 neurqo_current_search_k > 0 ?
		 neurqo_current_search_k : neurqo_search_topk,
		act.note, *policy_ms);
	if (state_json_out != NULL)
		*state_json_out = state_json;
	else
		pfree(state_json);
	return true;
}

static bool
neurqo_policy_low(Query* q,
				  int round, PlannedStmt* selected_plan,
				  double cumulative_cost_ms,
				  int max_split_rounds,
				  bool is_split_execution,
				  char** aja_hint_out, double* policy_ms,
				  char** state_json_out)
{
	NeurqoPolicyAction act;
	char* state_json;
	bool ok;

	neurqo_current_execution_action[0] = '\0';
	neurqo_current_lip_action[0] = '\0';
	*aja_hint_out = NULL;
	if (state_json_out != NULL)
		*state_json_out = NULL;
	state_json = neurqo_build_low_state(
		q, round, selected_plan,
		cumulative_cost_ms, max_split_rounds, is_split_execution);
	ok = neurqo_request_policy_action("low", round, state_json, &act,
									 policy_ms);
	if (!ok)
	{
		pfree(state_json);
		return false;
	}

	if (act.has_execution_action)
		snprintf(neurqo_current_execution_action,
				 sizeof(neurqo_current_execution_action),
				 "%s", act.execution_action);
	if (act.has_lip_action)
		snprintf(neurqo_current_lip_action,
				 sizeof(neurqo_current_lip_action),
				 "%s", act.lip_action);
	if (neurqo_adaptive_aja_level() == NULL &&
		act.has_aja_hint &&
		pg_strcasecmp(act.aja_hint, "none") != 0 &&
		pg_strcasecmp(act.aja_hint, "default") != 0)
	{
		if (strchr(act.aja_hint, '(') != NULL)
			*aja_hint_out = pstrdup(act.aja_hint);
		else
			*aja_hint_out = neurqo_build_join_method_hint(q, act.aja_hint);
	}
	else if (neurqo_adaptive_aja_level() == NULL &&
			 act.has_join_method &&
			 pg_strcasecmp(act.join_method, "none") != 0 &&
			 pg_strcasecmp(act.join_method, "default") != 0)
		*aja_hint_out = neurqo_build_join_method_hint(q, act.join_method);

	elog(LOG, "[neurqo] run=" UINT64_FORMAT
		 " round %d: low execution_action=%s lip_action=%s aja_hint=%s note=\"%s\" policy_ms=%.2f",
		 neurqo_current_run_id, round,
		 neurqo_aja_enabled() ? neurqo_current_execution_action : "none",
		 neurqo_lip_enabled() ? neurqo_current_lip_action : "none",
		*aja_hint_out != NULL ? *aja_hint_out : "none",
		act.note, *policy_ms);
	if (state_json_out != NULL)
		*state_json_out = state_json;
	else
		pfree(state_json);
	return true;
}

static void
neurqo_flatten_and_clauses(Node* node, List** clauses)
{
	if (node == NULL)
		return;
	if (IsA(node, BoolExpr) && ((BoolExpr*)node)->boolop == AND_EXPR)
	{
		ListCell* lc;

		foreach(lc, ((BoolExpr*)node)->args)
			neurqo_flatten_and_clauses((Node*)lfirst(lc), clauses);
		return;
	}
	*clauses = lappend(*clauses, node);
}

static bool
neurqo_clause_single_varno(Node* clause, Index* varno)
{
	List* vars;
	ListCell* lc;
	bool found = false;
	Index vno = 0;

	vars = pull_var_clause(clause, 0);
	foreach(lc, vars)
	{
		Var* var = (Var*)lfirst(lc);

		if (!IsA(var, Var) || var->varlevelsup != 0)
			continue;
		if (!found)
		{
			vno = var->varno;
			found = true;
		}
		else if (vno != var->varno)
		{
			list_free(vars);
			return false;
		}
	}
	list_free(vars);
	if (!found)
		return false;
	*varno = vno;
	return true;
}

static RangeTblEntry*
neurqo_rte_for_var(Query* q, Var* var)
{
	RangeTblEntry* rte;

	if (var == NULL || var->varlevelsup != 0 ||
		var->varno == 0 || var->varno > list_length(q->rtable))
		return NULL;

	rte = (RangeTblEntry*)list_nth(q->rtable, var->varno - 1);
	if (rte->rtekind != RTE_RELATION || rte->relkind != RELKIND_RELATION)
		return NULL;
	return rte;
}

static bool
neurqo_is_int4_equi_join(Expr* expr, Var** left, Var** right)
{
	OpExpr* op;
	Node* lnode;
	Node* rnode;
	Var* lvar;
	Var* rvar;

	if (expr == NULL || !IsA(expr, OpExpr))
		return false;
	op = (OpExpr*)expr;
	if (list_length(op->args) != 2)
		return false;
	lnode = (Node*)linitial(op->args);
	rnode = (Node*)lsecond(op->args);
	if (!IsA(lnode, Var) || !IsA(rnode, Var))
		return false;
	lvar = (Var*)lnode;
	rvar = (Var*)rnode;
	if (lvar->varlevelsup != 0 || rvar->varlevelsup != 0 ||
		lvar->varno == rvar->varno ||
		lvar->varattno <= 0 || rvar->varattno <= 0 ||
		lvar->vartype != INT4OID || rvar->vartype != INT4OID)
		return false;
	if (!op_hashjoinable(op->opno, INT4OID))
		return false;
	*left = lvar;
	*right = rvar;
	return true;
}

static bool
neurqo_var_att_is_id(Query* q, Var* var)
{
	RangeTblEntry* rte = neurqo_rte_for_var(q, var);
	char* attname;
	bool is_id;

	if (rte == NULL)
		return false;
	attname = get_attname(rte->relid, var->varattno, true);
	if (attname == NULL)
		return false;
	is_id = strcmp(attname, "id") == 0;
	pfree(attname);
	return is_id;
}

static bool
neurqo_lip_filter_exists(NeurqoLipFilter* filters, int nfilters,
						 Var* build_var, Var* probe_var)
{
	int i;

	for (i = 0; i < nfilters; i++)
	{
		if (filters[i].build_var->varno == build_var->varno &&
			filters[i].build_var->varattno == build_var->varattno &&
			filters[i].probe_var->varno == probe_var->varno &&
			filters[i].probe_var->varattno == probe_var->varattno)
			return true;
	}
	return false;
}

static Index
neurqo_lip_scan_relid(Plan* plan)
{
	if (plan == NULL)
		return 0;
	if (IsA(plan, SeqScan) ||
		IsA(plan, SampleScan) ||
		IsA(plan, IndexScan) ||
		IsA(plan, IndexOnlyScan) ||
		IsA(plan, BitmapHeapScan) ||
		IsA(plan, TidScan) ||
		IsA(plan, TidRangeScan) ||
		IsA(plan, ForeignScan) ||
		IsA(plan, CustomScan))
		return ((Scan*)plan)->scanrelid;
	return 0;
}

static void
neurqo_lip_plan_rel_stats(Plan* plan, Index varno,
						  NeurqoLipRelStats* stats)
{
	Index scanrelid;

	if (plan == NULL)
		return;
	scanrelid = neurqo_lip_scan_relid(plan);
	if (scanrelid == varno)
	{
		stats->has_plan = true;
		stats->plan_rows += Max(plan->plan_rows, 0.0);
		stats->plan_cost += Max(plan->total_cost, 0.0);
	}
	neurqo_lip_plan_rel_stats(plan->lefttree, varno, stats);
	neurqo_lip_plan_rel_stats(plan->righttree, varno, stats);
}

static double
neurqo_lip_relation_rows(Query* q, Var* var)
{
	RangeTblEntry* rte = neurqo_rte_for_var(q, var);
	HeapTuple tuple;
	Form_pg_class classform;
	double rows = -1.0;

	if (rte == NULL)
		return rows;
	tuple = SearchSysCache1(RELOID, ObjectIdGetDatum(rte->relid));
	if (!HeapTupleIsValid(tuple))
		return rows;
	classform = (Form_pg_class)GETSTRUCT(tuple);
	rows = classform->reltuples;
	ReleaseSysCache(tuple);
	return rows;
}

static NeurqoLipRelStats
neurqo_lip_rel_stats(Query* q, PlannedStmt* reference_plan, Var* var)
{
	NeurqoLipRelStats stats;

	memset(&stats, 0, sizeof(stats));
	stats.relation_rows = neurqo_lip_relation_rows(q, var);
	if (reference_plan != NULL)
		neurqo_lip_plan_rel_stats(reference_plan->planTree,
								  var->varno, &stats);
	if (stats.relation_rows < 0.0 && stats.has_plan)
		stats.relation_rows = stats.plan_rows;
	return stats;
}

static bool
neurqo_lip_build_eligible(NeurqoLipRelStats* build,
						  NeurqoLipRelStats* probe,
						  bool selective)
{
	if (build->relation_rows < 0.0 ||
		build->relation_rows >
			(double)neurqo_lip_max_build_relation_rows)
		return false;
	if (selective &&
			(!build->has_plan ||
			 build->plan_rows > (double)neurqo_lip_selective_plan_rows ||
			 build->relation_rows <= 0.0 ||
			 build->plan_rows * 100.0 >
				build->relation_rows *
				(double)neurqo_lip_max_build_selectivity_pct))
		return false;
	if (build->has_plan && probe->has_plan &&
		probe->plan_rows <
			build->plan_rows * (double)neurqo_lip_min_probe_ratio)
		return false;
	return true;
}

static int
neurqo_lip_existing_build_filter(NeurqoLipFilter* filters, int nprobes,
								 Var* build_var)
{
	int i;

	for (i = 0; i < nprobes; i++)
	{
		if (filters[i].build_var->varno == build_var->varno &&
			filters[i].build_var->varattno == build_var->varattno)
			return filters[i].filter_id;
	}
	return -1;
}

static int
neurqo_collect_lip_filters(Query* q, List* clauses,
						   PlannedStmt* reference_plan,
						   NeurqoLipFilter* filters,
						   int* filter_count)
{
	bool* has_local_restrict;
	ListCell* lc;
	int nprobes = 0;
	int nfilters = 0;
	int nrtables = list_length(q->rtable);
	bool selective = strcmp(neurqo_current_lip_action, "selective") == 0;

	has_local_restrict = (bool*)palloc0((nrtables + 1) * sizeof(bool));
	foreach(lc, clauses)
	{
		Index varno;

		if (neurqo_clause_single_varno((Node*)lfirst(lc), &varno) &&
			varno > 0 && varno <= nrtables)
			has_local_restrict[varno] = true;
	}

	foreach(lc, clauses)
	{
		Var* left = NULL;
		Var* right = NULL;
		Var* build_var;
		Var* probe_var;
		bool left_local;
		bool right_local;
		bool left_eligible;
		bool right_eligible;
		NeurqoLipRelStats left_stats;
		NeurqoLipRelStats right_stats;
		int filter_id;

		if (!neurqo_is_int4_equi_join((Expr*)lfirst(lc), &left, &right))
			continue;
		if (neurqo_rte_for_var(q, left) == NULL ||
			neurqo_rte_for_var(q, right) == NULL)
			continue;

		left_local = left->varno <= nrtables && has_local_restrict[left->varno];
		right_local = right->varno <= nrtables && has_local_restrict[right->varno];
		if (!left_local && !right_local)
			continue;
		left_stats = neurqo_lip_rel_stats(q, reference_plan, left);
		right_stats = neurqo_lip_rel_stats(q, reference_plan, right);
		left_eligible = left_local &&
			neurqo_lip_build_eligible(&left_stats, &right_stats, selective);
		right_eligible = right_local &&
			neurqo_lip_build_eligible(&right_stats, &left_stats, selective);
		if (!left_eligible && !right_eligible)
			continue;

		if (right_eligible && !left_eligible)
		{
			build_var = right;
			probe_var = left;
		}
		else if (left_eligible && !right_eligible)
		{
			build_var = left;
			probe_var = right;
		}
		else if ((right_stats.has_plan && left_stats.has_plan &&
				  right_stats.plan_rows < left_stats.plan_rows) ||
				 (right_stats.plan_rows == left_stats.plan_rows &&
				  right_stats.relation_rows < left_stats.relation_rows) ||
				 (right_stats.plan_rows == left_stats.plan_rows &&
				  right_stats.relation_rows == left_stats.relation_rows &&
				  !neurqo_var_att_is_id(q, left) &&
				  neurqo_var_att_is_id(q, right)))
		{
			build_var = right;
			probe_var = left;
		}
		else
		{
			build_var = left;
			probe_var = right;
		}

		if (neurqo_lip_filter_exists(filters, nprobes, build_var, probe_var))
			continue;
		filter_id = neurqo_lip_existing_build_filter(
			filters, nprobes, build_var);
		if (filter_id < 0)
		{
			if (nfilters >= Min(neurqo_lip_max_filters,
							   NEURQO_MAX_LIP_FILTERS))
				continue;
			filter_id = nfilters++;
			filters[nprobes].is_build = true;
		}
		else
			filters[nprobes].is_build = false;
		filters[nprobes].filter_id = filter_id;
		filters[nprobes].build_var = (Var*)copyObjectImpl(build_var);
		filters[nprobes].probe_var = (Var*)copyObjectImpl(probe_var);
		nprobes++;
		if (nprobes >= NEURQO_MAX_LIP_PROBES)
			break;
	}

	pfree(has_local_restrict);
	*filter_count = nfilters;
	return nprobes;
}

static Node*
neurqo_remap_single_varno_mutator(Node* node, void* context)
{
	NeurqoVarnoRemapContext* ctx = (NeurqoVarnoRemapContext*)context;

	if (node == NULL)
		return NULL;
	if (IsA(node, Var))
	{
		Var* oldvar = (Var*)node;
		Var* newvar = (Var*)copyObjectImpl(oldvar);

		if (newvar->varlevelsup == 0 && newvar->varno == ctx->from_varno)
		{
			newvar->varno = ctx->to_varno;
			newvar->varnosyn = ctx->to_varno;
		}
		return (Node*)newvar;
	}
	return expression_tree_mutator(node, neurqo_remap_single_varno_mutator,
								   context);
}

static char*
neurqo_lip_deparse_where(Query* q, List* clauses, Var* build_var)
{
	RangeTblEntry* rte = neurqo_rte_for_var(q, build_var);
	StringInfoData where;
	ListCell* lc;

	if (rte == NULL)
		return NULL;
	initStringInfo(&where);
	foreach(lc, clauses)
	{
		Node* clause = (Node*)lfirst(lc);
		Index varno;
		NeurqoVarnoRemapContext ctx;
		Node* local_clause;
		List* dpcontext;
		char* clause_sql;

		if (!neurqo_clause_single_varno(clause, &varno) ||
			varno != build_var->varno)
			continue;
		ctx.from_varno = build_var->varno;
		ctx.to_varno = 1;
		local_clause = neurqo_remap_single_varno_mutator(clause, &ctx);
		dpcontext = deparse_context_for(rte->eref->aliasname, rte->relid);
		clause_sql = deparse_expression(local_clause, dpcontext, false, false);
		if (where.len > 0)
			appendStringInfoString(&where, " AND ");
		appendStringInfo(&where, "(%s)", clause_sql);
	}
	if (where.len == 0)
	{
		pfree(where.data);
		return NULL;
	}
	return where.data;
}

static bool
neurqo_lip_execute_sql(const char* sql)
{
	int rc = SPI_execute(sql, false, 0);

	if (rc < 0)
	{
		elog(WARNING, "[neurqo] run=" UINT64_FORMAT " LIP SQL failed rc=%d sql=%s",
			 neurqo_current_run_id, rc, sql);
		return false;
	}
	return true;
}

static bool
neurqo_lip_run_setup(Query* q, List* clauses, NeurqoLipFilter* filters,
					 int nprobes, int nfilters)
{
	int i;
	int spi_rc;
	bool pushed_snapshot = false;
	int save_client_min_messages = client_min_messages;

	client_min_messages = WARNING;
	if (!ActiveSnapshotSet())
	{
		PushActiveSnapshot(GetTransactionSnapshot());
		pushed_snapshot = true;
	}

	spi_rc = SPI_connect();
	if (spi_rc != SPI_OK_CONNECT)
	{
		elog(WARNING, "[neurqo] run=" UINT64_FORMAT " LIP setup skipped: SPI_connect rc=%d",
			 neurqo_current_run_id, spi_rc);
		if (pushed_snapshot)
			PopActiveSnapshot();
		client_min_messages = save_client_min_messages;
		return false;
	}

	if (!neurqo_lip_execute_sql("CREATE EXTENSION IF NOT EXISTS pg_lip_bloom"))
		goto fail;
	CommandCounterIncrement();
	if (pushed_snapshot)
		UpdateActiveSnapshotCommandId();
	if (!neurqo_lip_execute_sql("SELECT pg_lip_bloom_set_dynamic(2)"))
		goto fail;
	if (!neurqo_lip_execute_sql(psprintf("SELECT pg_lip_bloom_init(%d)", nfilters)))
		goto fail;

	for (i = 0; i < nprobes; i++)
	{
		RangeTblEntry* rte = neurqo_rte_for_var(q, filters[i].build_var);
		char* schema;
		char* relname;
		char* attname;
		char* relation_sql;
		const char* alias_sql;
		const char* att_sql;
		char* where_sql;
		StringInfoData sql;

		if (!filters[i].is_build)
			continue;
		if (rte == NULL)
			goto fail;
		schema = get_namespace_name(get_rel_namespace(rte->relid));
		relname = get_rel_name(rte->relid);
		attname = get_attname(rte->relid, filters[i].build_var->varattno, true);
		if (schema == NULL || relname == NULL || attname == NULL)
			goto fail;
		relation_sql = quote_qualified_identifier(schema, relname);
		alias_sql = quote_identifier(rte->eref->aliasname);
		att_sql = quote_identifier(attname);
		where_sql = neurqo_lip_deparse_where(q, clauses, filters[i].build_var);

		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT sum(pg_lip_bloom_add(%d, %s.%s)) FROM %s AS %s",
						 filters[i].filter_id, alias_sql, att_sql,
						 relation_sql, alias_sql);
		if (where_sql != NULL)
			appendStringInfo(&sql, " WHERE %s", where_sql);
		elog(LOG, "[neurqo] run=" UINT64_FORMAT " LIP build filter=%d sql=%s",
			 neurqo_current_run_id, filters[i].filter_id, sql.data);
		if (!neurqo_lip_execute_sql(sql.data))
			goto fail;
		pfree(sql.data);
	}

	SPI_finish();
	CommandCounterIncrement();
	if (pushed_snapshot)
		PopActiveSnapshot();
	client_min_messages = save_client_min_messages;
	return true;

fail:
	SPI_finish();
	if (pushed_snapshot)
		PopActiveSnapshot();
	client_min_messages = save_client_min_messages;
	return false;
}

static Oid
neurqo_lip_probe_funcid(void)
{
	Oid argtypes[2] = {INT4OID, INT4OID};

	return LookupFuncName(list_make1(makeString("pg_lip_bloom_probe")),
						  2, argtypes, true);
}

static void
neurqo_lip_add_probe_qual(Query* q, Oid probe_funcid, NeurqoLipFilter* filter)
{
	Const* filter_id;
	FuncExpr* probe;
	Node* old_quals;

	filter_id = makeConst(INT4OID, -1, InvalidOid, sizeof(int32),
						  Int32GetDatum(filter->filter_id), false, true);
	probe = makeFuncExpr(probe_funcid, BOOLOID,
						 list_make2(filter_id, copyObjectImpl(filter->probe_var)),
						 InvalidOid, InvalidOid, COERCE_EXPLICIT_CALL);
	old_quals = q->jointree->quals;
	if (old_quals == NULL)
		q->jointree->quals = (Node*)probe;
	else if (IsA(old_quals, BoolExpr) &&
			 ((BoolExpr*)old_quals)->boolop == AND_EXPR)
		((BoolExpr*)old_quals)->args =
			lappend(((BoolExpr*)old_quals)->args, probe);
	else
		q->jointree->quals =
			(Node*)makeBoolExpr(AND_EXPR, list_make2(old_quals, probe), -1);
}

static bool
neurqo_apply_lip(Query* q, PlannedStmt* reference_plan,
				 double* lip_ms, int* lip_filters)
{
	List* clauses = NIL;
	NeurqoLipFilter filters[NEURQO_MAX_LIP_PROBES];
	Oid probe_funcid;
	int nprobes;
	int nfilters;
	int i;
	double t0 = neurqo_now_ms();

	*lip_ms = 0.0;
	*lip_filters = 0;
	if (!neurqo_lip_enabled() || q == NULL || q->jointree == NULL)
		return true;

	neurqo_flatten_and_clauses(q->jointree->quals, &clauses);
	nprobes = neurqo_collect_lip_filters(q, clauses, reference_plan,
										filters, &nfilters);
	if (nfilters <= 0)
	{
		elog(LOG, "[neurqo] run=" UINT64_FORMAT " LIP skipped: no eligible int4 equi-join filters mode=%s",
			 neurqo_current_run_id, neurqo_current_lip_action);
		return true;
	}

	if (!neurqo_lip_run_setup(q, clauses, filters, nprobes, nfilters))
	{
		elog(WARNING, "[neurqo] run=" UINT64_FORMAT " LIP setup failed; continuing without probe quals",
			 neurqo_current_run_id);
		return false;
	}

	probe_funcid = neurqo_lip_probe_funcid();
	if (!OidIsValid(probe_funcid))
	{
		elog(WARNING, "[neurqo] run=" UINT64_FORMAT " LIP probe function not found after setup; continuing without probe quals",
			 neurqo_current_run_id);
		return false;
	}

	for (i = 0; i < nprobes; i++)
		neurqo_lip_add_probe_qual(q, probe_funcid, &filters[i]);

	*lip_ms = neurqo_now_ms() - t0;
	*lip_filters = nfilters;
	elog(LOG, "[neurqo] run=" UINT64_FORMAT
		 " apply LIP: mode=%s filters=%d probes=%d lip_ms=%.2f",
		 neurqo_current_run_id, neurqo_current_lip_action,
		 nfilters, nprobes, *lip_ms);
	return true;
}

/*
 * PG16 moved per-RTE permission data out of RangeTblEntry into a separate
 * query->rteperminfos list, indexed by rte->perminfoindex.  The original
 * (PG12) querysplit builds/remaps Query trees by hand and never maintains
 * that list, so the executor dereferences a stale/empty perminfo and crashes
 * (SIGSEGV).  neurqo_rebuild_perminfos() regenerates rteperminfos to match the
 * current (subset/remapped) rtable; neurqo_plan() does it right before every
 * planner() call so every plan/exec sees consistent permission info.
 */
static void
neurqo_rebuild_perminfos(Query* q)
{
	List* old = q->rteperminfos;
	ListCell* lc;
	q->rteperminfos = NIL;
	foreach(lc, q->rtable)
	{
		RangeTblEntry* rte = (RangeTblEntry*) lfirst(lc);
		int old_idx = rte->perminfoindex;
		rte->perminfoindex = 0;		/* addRTEPermissionInfo asserts == 0 */
		if (rte->rtekind == RTE_RELATION && old_idx != 0)
		{
			RTEPermissionInfo* npi = addRTEPermissionInfo(&q->rteperminfos, rte);
			if (old != NIL && old_idx <= list_length(old))
			{
				RTEPermissionInfo* opi = (RTEPermissionInfo*) list_nth(old, old_idx - 1);
				npi->inh = opi->inh;
				npi->requiredPerms = opi->requiredPerms;
				npi->checkAsUser = opi->checkAsUser;
				npi->selectedCols = opi->selectedCols;
				npi->insertedCols = opi->insertedCols;
				npi->updatedCols = opi->updatedCols;
			}
			else
				npi->requiredPerms = ACL_SELECT;
		}
	}
}

static PlannedStmt*
neurqo_plan_direct(Query* q, int cursorOptions, bool apply_lip,
				   const char* hint_query_string, bool log_hint)
{
	PlannedStmt* r;
	double lip_ms = 0.0;
	int lip_filters = 0;

	elog(DEBUG1, "[neurqo] run=" UINT64_FORMAT " plan: rtable=%d perminfos=%d",
		 neurqo_current_run_id, list_length(q->rtable), list_length(q->rteperminfos));
	if (apply_lip)
		neurqo_apply_lip(q, NULL, &lip_ms, &lip_filters);
	neurqo_rebuild_perminfos(q);
	elog(DEBUG1, "[neurqo] run=" UINT64_FORMAT " plan: perminfos rebuilt=%d, calling planner",
		 neurqo_current_run_id, list_length(q->rteperminfos));
	if (hint_query_string != NULL && log_hint)
		elog(LOG, "[neurqo] run=" UINT64_FORMAT " apply planner hint: search_strategy=%s execution_action=%s lip_action=%s hint=%s",
			 neurqo_current_run_id,
			 neurqo_search_enabled() ? neurqo_current_search_strategy : "default",
			 neurqo_aja_enabled() ? neurqo_current_execution_action : "none",
			 neurqo_lip_enabled() ? neurqo_current_lip_action : "none",
			 hint_query_string);
	r = planner(q, hint_query_string, cursorOptions, NULL);
	elog(DEBUG1, "[neurqo] run=" UINT64_FORMAT " plan: planner returned ok lip_filters=%d lip_ms=%.2f",
		 neurqo_current_run_id, lip_filters, lip_ms);
	return r;
}

static PlannedStmt*
neurqo_plan(Query* q, int cursorOptions, bool apply_lip)
{
	PlannedStmt* r;
	char* hint_query_string = NULL;

	hint_query_string = neurqo_build_planner_hint(q);
	if (apply_lip)
	{
		double lip_ms = 0.0;
		int lip_filters = 0;

		neurqo_apply_lip(q, NULL, &lip_ms, &lip_filters);
	}
	r = neurqo_plan_direct(q, cursorOptions, false, hint_query_string, true);
	if (hint_query_string != NULL)
		pfree(hint_query_string);
	return r;
}

static PlannedStmt*
neurqo_plan_nestloop_candidate(Query* q, const char* hint_query_string)
{
	PlannedStmt* plan = NULL;
	bool saved_nestloop = enable_nestloop;
	bool saved_mergejoin = enable_mergejoin;
	bool saved_hashjoin = enable_hashjoin;

	PG_TRY();
	{
		enable_nestloop = true;
		enable_mergejoin = false;
		enable_hashjoin = false;
		plan = neurqo_plan_direct(q, 0, false, hint_query_string, false);
	}
	PG_FINALLY();
	{
		enable_nestloop = saved_nestloop;
		enable_mergejoin = saved_mergejoin;
		enable_hashjoin = saved_hashjoin;
	}
	PG_END_TRY();

	return plan;
}

static PlannedStmt*
neurqo_plan_hashjoin_candidate(Query* q, const char* hint_query_string)
{
	PlannedStmt* plan = NULL;
	bool saved_nestloop = enable_nestloop;
	bool saved_mergejoin = enable_mergejoin;
	bool saved_hashjoin = enable_hashjoin;

	PG_TRY();
	{
		enable_nestloop = false;
		enable_mergejoin = false;
		enable_hashjoin = true;
		plan = neurqo_plan_direct(q, 0, false, hint_query_string, false);
	}
	PG_FINALLY();
	{
		enable_nestloop = saved_nestloop;
		enable_mergejoin = saved_mergejoin;
		enable_hashjoin = saved_hashjoin;
	}
	PG_END_TRY();

	return plan;
}

static char*
neurqo_make_hint_query(const char* first_hint, const char* second_hint)
{
	StringInfoData hint;

	if ((first_hint == NULL || first_hint[0] == '\0') &&
		(second_hint == NULL || second_hint[0] == '\0'))
		return NULL;

	initStringInfo(&hint);
	appendStringInfoString(&hint, "/*+");
	if (first_hint != NULL && first_hint[0] != '\0')
		appendStringInfo(&hint, " %s", first_hint);
	if (second_hint != NULL && second_hint[0] != '\0')
		appendStringInfo(&hint, " %s", second_hint);
	appendStringInfoString(&hint, " */\nSELECT 1");
	return hint.data;
}

static char*
neurqo_build_left_deep_leading_hint(Query* q)
{
	ListCell* lc;
	StringInfoData leading;
	int nrels = 0;

	initStringInfo(&leading);
	foreach(lc, q->rtable)
	{
		RangeTblEntry* rte = (RangeTblEntry*)lfirst(lc);
		const char* alias;

		if (rte->rtekind != RTE_RELATION)
			continue;
		alias = rte->eref ? rte->eref->aliasname : NULL;
		if (alias == NULL || alias[0] == '\0')
			continue;
		if (nrels == 0)
			appendStringInfoString(&leading, alias);
		else if (nrels == 1)
		{
			char* prev = pstrdup(leading.data);

			resetStringInfo(&leading);
			appendStringInfo(&leading, "(%s %s)", prev, alias);
			pfree(prev);
		}
		else
		{
			char* prev = pstrdup(leading.data);

			resetStringInfo(&leading);
			appendStringInfo(&leading, "(%s %s)", prev, alias);
			pfree(prev);
		}
		nrels++;
	}

	if (nrels < 2)
	{
		pfree(leading.data);
		return NULL;
	}
	{
		char* ret = psprintf("Leading(%s)", leading.data);

		pfree(leading.data);
		return ret;
	}
}

static char*
neurqo_build_join_method_hint(Query* q, const char* method)
{
	ListCell* lc;
	StringInfoData aliases;
	const char* pg_hint_method = NULL;
	int nrels = 0;

	if (method == NULL || method[0] == '\0')
		return NULL;
	if (pg_strcasecmp(method, "hashjoin") == 0 ||
		pg_strcasecmp(method, "hash") == 0 ||
		pg_strcasecmp(method, "aja") == 0)
		pg_hint_method = "HashJoin";
	else if (pg_strcasecmp(method, "nestloop") == 0 ||
			 pg_strcasecmp(method, "nested_loop") == 0 ||
			 pg_strcasecmp(method, "nl") == 0)
		pg_hint_method = "NestLoop";
	else if (pg_strcasecmp(method, "mergejoin") == 0 ||
			 pg_strcasecmp(method, "merge") == 0)
		pg_hint_method = "MergeJoin";
	else if (pg_strcasecmp(method, "none") == 0 ||
			 pg_strcasecmp(method, "default") == 0)
		return NULL;
	else
		pg_hint_method = method;

	initStringInfo(&aliases);
	foreach(lc, q->rtable)
	{
		RangeTblEntry* rte = (RangeTblEntry*)lfirst(lc);
		const char* alias;

		if (rte->rtekind != RTE_RELATION)
			continue;
		alias = rte->eref ? rte->eref->aliasname : NULL;
		if (alias == NULL || alias[0] == '\0')
			continue;
		if (aliases.len > 0)
			appendStringInfoChar(&aliases, ' ');
		appendStringInfoString(&aliases, alias);
		nrels++;
	}
	if (nrels < 2)
	{
		pfree(aliases.data);
		return NULL;
	}
	{
		char* ret = psprintf("%s(%s)", pg_hint_method, aliases.data);

		pfree(aliases.data);
		return ret;
	}
}

static int
neurqo_effective_search_k(void)
{
	int k = neurqo_current_search_k > 0 ?
		neurqo_current_search_k : neurqo_search_topk;

	if (k <= 0)
		k = 1;
	if (k > NEURQO_SEARCH_ABS_MAX_K)
		k = NEURQO_SEARCH_ABS_MAX_K;
	return k;
}

static int
neurqo_popcount64(uint64 mask)
{
	int n = 0;

	while (mask != 0)
	{
		n += (mask & 1) ? 1 : 0;
		mask >>= 1;
	}
	return n;
}

static Var*
neurqo_node_var(Node* node)
{
	if (node == NULL)
		return NULL;
	if (IsA(node, Var))
		return (Var*)node;
	if (IsA(node, RelabelType))
	{
		Node* arg = (Node*)((RelabelType*)node)->arg;

		if (arg != NULL && IsA(arg, Var))
			return (Var*)arg;
	}
	return NULL;
}

static int
neurqo_collect_search_rels(Query* q, NeurqoSearchRel* rels, int maxrels,
						   bool* all_rte_relation)
{
	ListCell* lc;
	int rtindex = 0;
	int nrels = 0;

	*all_rte_relation = true;
	foreach(lc, q->rtable)
	{
		RangeTblEntry* rte = (RangeTblEntry*)lfirst(lc);

		rtindex++;
		if (rte->rtekind != RTE_RELATION)
		{
			*all_rte_relation = false;
			continue;
		}
		if (nrels >= maxrels)
			return nrels;
		rels[nrels].rtindex = rtindex;
		rels[nrels].alias = pstrdup(rte->eref && rte->eref->aliasname ?
									rte->eref->aliasname : get_rel_name(rte->relid));
		nrels++;
	}
	return nrels;
}

static void
neurqo_collect_join_edges(Query* q, NeurqoSearchRel* rels, int nrels,
						  bool* edges)
{
	int* rtindex_to_pos;
	List* clauses = NIL;
	ListCell* lc;
	int nrtables = list_length(q->rtable);
	int i;

	memset(edges, false, nrels * nrels * sizeof(bool));
	if (q->jointree == NULL || q->jointree->quals == NULL)
		return;

	rtindex_to_pos = (int*)palloc((nrtables + 1) * sizeof(int));
	for (i = 0; i <= nrtables; i++)
		rtindex_to_pos[i] = -1;
	for (i = 0; i < nrels; i++)
		rtindex_to_pos[rels[i].rtindex] = i;

	neurqo_flatten_and_clauses(q->jointree->quals, &clauses);
	foreach(lc, clauses)
	{
		Expr* expr = (Expr*)lfirst(lc);
		OpExpr* op;
		Var* left;
		Var* right;
		int lpos;
		int rpos;

		if (expr == NULL || !IsA(expr, OpExpr))
			continue;
		op = (OpExpr*)expr;
		if (list_length(op->args) != 2)
			continue;
		left = neurqo_node_var((Node*)linitial(op->args));
		right = neurqo_node_var((Node*)lsecond(op->args));
		if (left == NULL || right == NULL ||
			left->varlevelsup != 0 || right->varlevelsup != 0 ||
			left->varno == right->varno ||
			left->varno <= 0 || right->varno <= 0 ||
			left->varno > nrtables || right->varno > nrtables)
			continue;
		lpos = rtindex_to_pos[left->varno];
		rpos = rtindex_to_pos[right->varno];
		if (lpos < 0 || rpos < 0)
			continue;
		edges[lpos * nrels + rpos] = true;
		edges[rpos * nrels + lpos] = true;
	}
	pfree(rtindex_to_pos);
}

static bool
neurqo_masks_connected(uint64 lmask, uint64 rmask, bool* edges, int nrels)
{
	int i;
	int j;

	for (i = 0; i < nrels; i++)
	{
		if ((lmask & (((uint64)1) << i)) == 0)
			continue;
		for (j = 0; j < nrels; j++)
		{
			if ((rmask & (((uint64)1) << j)) == 0)
				continue;
			if (edges[i * nrels + j])
				return true;
		}
	}
	return false;
}

static Query*
neurqo_make_subset_query(Query* q, NeurqoSearchRel* rels, int nrels,
						 uint64 mask)
{
	int nrtables = list_length(q->rtable);
	Index* map = (Index*)palloc0(nrtables * sizeof(Index));
	List* local_rtable = NIL;
	int next = 1;
	int i;

	if (q->jointree == NULL || q->jointree->quals == NULL)
	{
		pfree(map);
		return NULL;
	}

	for (i = 0; i < nrels; i++)
	{
		if ((mask & (((uint64)1) << i)) != 0)
		{
			RangeTblEntry* rte = copyObjectImpl(list_nth(q->rtable,
														 rels[i].rtindex - 1));

			local_rtable = lappend(local_rtable, rte);
			map[rels[i].rtindex - 1] = next++;
		}
	}
	if (list_length(local_rtable) == 0)
	{
		pfree(map);
		return NULL;
	}
	{
		Query* local_query = createQuery(q, DestIntoRel, local_rtable, map,
										 nrtables);

		/*
		 * The subset plan estimates pre-aggregation join cardinality.  Keeping
		 * the parent GROUP/ORDER/DISTINCT metadata after createQuery() has
		 * replaced its target list leaves dangling sortgrouprefs (for example
		 * TPC-H Q16) and can make the planner fail before Search gets a chance
		 * to fall back.  Projection and upper-query operations do not belong
		 * in this estimate.
		 */
		local_query->targetList = NIL;
		local_query->returningList = NIL;
		local_query->groupClause = NIL;
		local_query->groupDistinct = false;
		local_query->groupingSets = NIL;
		local_query->havingQual = NULL;
		local_query->windowClause = NIL;
		local_query->distinctClause = NIL;
		local_query->sortClause = NIL;
		local_query->limitOffset = NULL;
		local_query->limitCount = NULL;
		local_query->rowMarks = NIL;
		local_query->setOperations = NULL;
		local_query->hasAggs = false;
		local_query->hasWindowFuncs = false;
		local_query->hasTargetSRFs = false;
		local_query->hasDistinctOn = false;
		local_query->hasForUpdate = false;

		pfree(map);
		return local_query;
	}
}

static double
neurqo_plan_subset_cardinality(Query* q, NeurqoSearchRel* rels, int nrels,
							   NeurqoSearchCell* cells, uint64 mask)
{
	Query* local_query;
	PlannedStmt* planned;
	double rows;

	if (cells[mask].card_valid)
		return cells[mask].card_rows;

	local_query = neurqo_make_subset_query(q, rels, nrels, mask);
	if (local_query == NULL)
		rows = 1.0;
	else
	{
		neurqo_last_search_planner_calls++;
		planned = neurqo_plan_direct(local_query, CURSOR_OPT_PARALLEL_OK,
									 false, NULL, false);
		rows = planned && planned->planTree ? planned->planTree->plan_rows : 1.0;
	}
	if (rows < 1.0)
		rows = 1.0;
	cells[mask].card_valid = true;
	cells[mask].card_rows = rows;
	return rows;
}

static double
neurqo_pairwise_subset_cardinality(Query* q, NeurqoSearchRel* rels,
								   int nrels, NeurqoSearchCell* cells,
								   bool* edges, uint64 mask)
{
	double log_rows = 0.0;
	double max_log_rows = log(DBL_MAX) - 1.0;
	int i;
	int j;

	if (cells[mask].card_valid)
		return cells[mask].card_rows;
	if (neurqo_popcount64(mask) <= 2)
		return neurqo_plan_subset_cardinality(q, rels, nrels, cells, mask);

	for (i = 0; i < nrels; i++)
	{
		uint64 singleton = ((uint64) 1) << i;
		double base_rows;

		if ((mask & singleton) == 0)
			continue;
		base_rows = neurqo_plan_subset_cardinality(
			q, rels, nrels, cells, singleton);
		log_rows += log(Max(base_rows, 1.0));
	}

	/*
	 * Compose the selectivity observed for every joined relation pair.  This
	 * is the same independence approximation used by a classical Selinger
	 * model, but it needs only O(|V|+|E|) PostgreSQL planning calls instead
	 * of one call for every connected subset.
	 */
	for (i = 0; i < nrels; i++)
	{
		uint64 imask = ((uint64) 1) << i;
		double left_rows;

		if ((mask & imask) == 0)
			continue;
		left_rows = cells[imask].card_rows;
		for (j = i + 1; j < nrels; j++)
		{
			uint64 jmask = ((uint64) 1) << j;
			uint64 pair_mask;
			double right_rows;
			double pair_rows;
			double selectivity;

			if ((mask & jmask) == 0 || !edges[i * nrels + j])
				continue;
			right_rows = cells[jmask].card_rows;
			pair_mask = imask | jmask;
			pair_rows = neurqo_plan_subset_cardinality(
				q, rels, nrels, cells, pair_mask);
			selectivity = pair_rows /
				Max(left_rows * right_rows, 1.0);
			selectivity = Min(1.0, Max(selectivity,
										 1.0 /
										 Max(left_rows * right_rows, 1.0)));
			log_rows += log(selectivity);
		}
	}

	cells[mask].card_valid = true;
	cells[mask].card_rows =
		log_rows >= max_log_rows ? DBL_MAX / 2.0 :
		Max(1.0, exp(log_rows));
	return cells[mask].card_rows;
}

static double
neurqo_subset_cardinality(Query* q, NeurqoSearchRel* rels, int nrels,
						  NeurqoSearchCell* cells, bool* edges, uint64 mask)
{
	if (neurqo_search_exact_cardinality)
		return neurqo_plan_subset_cardinality(
			q, rels, nrels, cells, mask);
	return neurqo_pairwise_subset_cardinality(
		q, rels, nrels, cells, edges, mask);
}

static void
neurqo_search_cell_add(NeurqoSearchCell* cell, double cout,
					   const char* leading, int k)
{
	int pos;
	int i;

	for (i = 0; i < cell->nentries; i++)
	{
		if (strcmp(cell->entries[i].leading, leading) == 0)
		{
			if (cout >= cell->entries[i].cout)
				return;
			cell->entries[i].cout = cout;
			break;
		}
	}
	if (i == cell->nentries)
	{
		if (cell->nentries >= k && cout >= cell->entries[cell->nentries - 1].cout)
			return;
		if (cell->nentries >= k)
		{
			pfree(cell->entries[cell->nentries - 1].leading);
			cell->nentries--;
		}
		cell->entries[cell->nentries].cout = cout;
		cell->entries[cell->nentries].leading = pstrdup(leading);
		cell->nentries++;
	}

	for (pos = 0; pos < cell->nentries; pos++)
	{
		int best = pos;

		for (i = pos + 1; i < cell->nentries; i++)
		{
			if (cell->entries[i].cout < cell->entries[best].cout)
				best = i;
		}
		if (best != pos)
		{
			NeurqoSearchEntry tmp = cell->entries[pos];

			cell->entries[pos] = cell->entries[best];
			cell->entries[best] = tmp;
		}
	}
}

static char*
neurqo_build_topk_leading_hint(Query* q, PlannedStmt** selected_plan_out)
{
	NeurqoSearchRel rels[NEURQO_SEARCH_ABS_MAX_RELS];
	bool edges[NEURQO_SEARCH_ABS_MAX_RELS * NEURQO_SEARCH_ABS_MAX_RELS];
	bool all_rte_relation;
	int nrels;
	int max_rels = neurqo_search_max_rels;
	int k = neurqo_effective_search_k();
	uint64 nmasks;
	uint64 full_mask;
	NeurqoSearchCell* cells;
	int level;
	int i;
	char* best_leading = NULL;
	double best_cost = DBL_MAX;
	double t0 = neurqo_now_ms();
	PlannedStmt* best_plan = NULL;

	if (selected_plan_out != NULL)
		*selected_plan_out = NULL;

	neurqo_last_search_ms = 0.0;
	neurqo_last_search_candidate_cost = 0.0;
	neurqo_last_search_candidates = 0;
	neurqo_last_search_planner_calls = 0;
	neurqo_last_search_applied = false;

	/*
	 * A SubLink can contain correlated Vars whose outer relation disappears
	 * from a DP subset.  Planning that synthetic subset is not semantically
	 * valid and PostgreSQL can fail with a lateral-reference error.  Keep the
	 * complete query on the native search path until subset construction can
	 * preserve correlated parameterization explicitly.
	 */
	if (q->hasSubLinks)
	{
		neurqo_last_search_ms = neurqo_now_ms() - t0;
		elog(LOG, "[neurqo] run=" UINT64_FORMAT
			 " Search top-k skipped: query contains SubLink; fallback default",
			 neurqo_current_run_id);
		return NULL;
	}

	if (max_rels <= 0 || max_rels > NEURQO_SEARCH_ABS_MAX_RELS)
		max_rels = NEURQO_SEARCH_ABS_MAX_RELS;
	nrels = neurqo_collect_search_rels(q, rels, NEURQO_SEARCH_ABS_MAX_RELS,
									   &all_rte_relation);
	if (nrels < 2)
		return NULL;
	if (!all_rte_relation || nrels > max_rels)
	{
		neurqo_last_search_ms = neurqo_now_ms() - t0;
		elog(LOG, "[neurqo] run=" UINT64_FORMAT " Search top-k skipped: nrels=%d all_relation=%d max_rels=%d; fallback default",
			 neurqo_current_run_id, nrels, all_rte_relation ? 1 : 0, max_rels);
		return NULL;
	}

	neurqo_collect_join_edges(q, rels, nrels, edges);
	full_mask = (((uint64)1) << nrels) - 1;
	nmasks = full_mask + 1;
	cells = (NeurqoSearchCell*)palloc0(sizeof(NeurqoSearchCell) * nmasks);

	for (i = 0; i < nrels; i++)
	{
		uint64 mask = ((uint64)1) << i;

		neurqo_search_cell_add(&cells[mask], 0.0, rels[i].alias, k);
		if (!neurqo_search_exact_cardinality)
			(void) neurqo_plan_subset_cardinality(
				q, rels, nrels, cells, mask);
	}

	for (level = 2; level <= nrels; level++)
	{
		uint64 mask;

		for (mask = 1; mask <= full_mask; mask++)
		{
			uint64 lmask;

			if (neurqo_popcount64(mask) != level)
				continue;
			for (lmask = (mask - 1) & mask; lmask != 0;
				 lmask = (lmask - 1) & mask)
			{
				uint64 rmask = mask ^ lmask;
				int li;
				int ri;
				double join_rows;

				if (rmask == 0 || lmask > rmask)
					continue;
				if (cells[lmask].nentries == 0 || cells[rmask].nentries == 0)
					continue;
				if (!neurqo_masks_connected(lmask, rmask, edges, nrels))
					continue;
				join_rows = neurqo_subset_cardinality(
					q, rels, nrels, cells, edges, mask);
				for (li = 0; li < cells[lmask].nentries; li++)
				{
					for (ri = 0; ri < cells[rmask].nentries; ri++)
					{
						double cout = cells[lmask].entries[li].cout +
							cells[rmask].entries[ri].cout + join_rows;
						char* leading;

						leading = psprintf("(%s %s)",
										   cells[lmask].entries[li].leading,
										   cells[rmask].entries[ri].leading);
						neurqo_search_cell_add(&cells[mask], cout, leading, k);
						pfree(leading);
						leading = psprintf("(%s %s)",
										   cells[rmask].entries[ri].leading,
										   cells[lmask].entries[li].leading);
						neurqo_search_cell_add(&cells[mask], cout, leading, k);
						pfree(leading);
					}
				}
			}
		}
	}

	if (cells[full_mask].nentries == 0)
	{
		neurqo_last_search_ms = neurqo_now_ms() - t0;
		elog(LOG, "[neurqo] run=" UINT64_FORMAT " Search top-k found no connected DP order; fallback default",
			 neurqo_current_run_id);
		return NULL;
	}
	neurqo_last_search_candidates = cells[full_mask].nentries;

	for (i = 0; i < cells[full_mask].nentries; i++)
	{
		char* search_hint = psprintf("Leading(%s)", cells[full_mask].entries[i].leading);
		char* hint_query = neurqo_make_hint_query(search_hint, NULL);
		PlannedStmt* planned;
		double cost;

		neurqo_last_search_planner_calls++;
		planned = neurqo_plan_direct(copyObjectImpl(q),
									CURSOR_OPT_PARALLEL_OK,
									false, hint_query, false);
		cost = planned && planned->planTree ?
			planned->planTree->total_cost : DBL_MAX;
		elog(DEBUG1, "[neurqo] run=" UINT64_FORMAT " Search candidate %d/%d cout=%.2f physical_cost=%.2f hint=%s",
			 neurqo_current_run_id, i + 1, cells[full_mask].nentries,
			 cells[full_mask].entries[i].cout, cost, search_hint);
		if (cost < best_cost)
		{
			best_cost = cost;
			if (best_leading != NULL)
				pfree(best_leading);
			best_leading = pstrdup(cells[full_mask].entries[i].leading);
			best_plan = planned;
		}
		pfree(search_hint);
		if (hint_query != NULL)
			pfree(hint_query);
	}

	neurqo_last_search_ms = neurqo_now_ms() - t0;
	if (best_cost < DBL_MAX)
		neurqo_last_search_candidate_cost = best_cost;
	if (best_leading == NULL)
		return NULL;

	elog(LOG, "[neurqo] run=" UINT64_FORMAT " Search top-k applied: strategy=%s k=%d nrels=%d candidates=%d best_physical_cost=%.2f search_ms=%.2f leading=%s",
		 neurqo_current_run_id, neurqo_current_search_strategy, k, nrels,
		 cells[full_mask].nentries, best_cost, neurqo_last_search_ms,
		 best_leading);
	neurqo_last_search_applied = true;
	if (selected_plan_out != NULL)
		*selected_plan_out = best_plan;
	{
		char* result = psprintf("Leading(%s)", best_leading);

		pfree(best_leading);
		return result;
	}
}

static char*
neurqo_build_search_hint(Query* q, PlannedStmt** selected_plan_out)
{
	if (selected_plan_out != NULL)
		*selected_plan_out = NULL;
	if (!neurqo_search_enabled())
		return NULL;
	if (strcmp(neurqo_current_search_strategy, "left_deep") == 0)
		return neurqo_build_left_deep_leading_hint(q);
	return neurqo_build_topk_leading_hint(q, selected_plan_out);
}

static const char*
neurqo_plan_node_name(Plan* plan)
{
	if (plan == NULL)
		return "Null";
	switch (nodeTag(plan))
	{
		case T_NestLoop:
			return "NestLoop";
		case T_MergeJoin:
			return "MergeJoin";
		case T_HashJoin:
			return "HashJoin";
		case T_SeqScan:
			return "SeqScan";
		case T_IndexScan:
			return "IndexScan";
		case T_IndexOnlyScan:
			return "IndexOnlyScan";
		case T_BitmapHeapScan:
			return "BitmapHeapScan";
		case T_BitmapIndexScan:
			return "BitmapIndexScan";
		case T_TidScan:
			return "TidScan";
		case T_SubqueryScan:
			return "SubqueryScan";
		case T_FunctionScan:
			return "FunctionScan";
		case T_ValuesScan:
			return "ValuesScan";
		case T_CteScan:
			return "CteScan";
		case T_Material:
			return "Material";
		case T_Sort:
			return "Sort";
		case T_Agg:
			return "Agg";
		case T_Group:
			return "Group";
		case T_Result:
			return "Result";
		case T_Limit:
			return "Limit";
		case T_Append:
			return "Append";
		case T_MergeAppend:
			return "MergeAppend";
		case T_Gather:
			return "Gather";
		case T_GatherMerge:
			return "GatherMerge";
		case T_Hash:
			return "Hash";
		default:
			return "Other";
	}
}

static bool
neurqo_plan_is_join(Plan* plan)
{
	return plan != NULL &&
		(IsA(plan, NestLoop) || IsA(plan, MergeJoin) || IsA(plan, HashJoin));
}

static bool
neurqo_plan_is_scan(Plan* plan)
{
	if (plan == NULL)
		return false;
	switch (nodeTag(plan))
	{
		case T_SeqScan:
		case T_IndexScan:
		case T_IndexOnlyScan:
		case T_BitmapHeapScan:
		case T_BitmapIndexScan:
		case T_TidScan:
		case T_SubqueryScan:
		case T_FunctionScan:
		case T_ValuesScan:
		case T_CteScan:
			return true;
		default:
			return false;
	}
}

static Index
neurqo_plan_scanrelid(Plan* plan)
{
	if (plan == NULL)
		return 0;
	switch (nodeTag(plan))
	{
		case T_SeqScan:
		case T_IndexScan:
		case T_IndexOnlyScan:
		case T_BitmapHeapScan:
		case T_TidScan:
		case T_SubqueryScan:
			return ((Scan*)plan)->scanrelid;
		default:
			return 0;
	}
}

static bool
neurqo_plan_hint_part(Plan* plan, Query* q, StringInfo methods,
						  char** tree_out, char** aliases_out,
						  bool include_methods, bool swap_hash_inputs)
{
	Index scanrelid;

	*tree_out = NULL;
	*aliases_out = NULL;
	if (plan == NULL)
		return false;

	scanrelid = neurqo_plan_scanrelid(plan);
	if (scanrelid > 0 && scanrelid <= list_length(q->rtable))
	{
		RangeTblEntry* rte =
			(RangeTblEntry*)list_nth(q->rtable, scanrelid - 1);
		const char* alias =
			rte->eref != NULL ? rte->eref->aliasname : NULL;
		const char* scan_method = NULL;

		if (alias == NULL || alias[0] == '\0')
			return false;
		if (include_methods)
		{
			if (IsA(plan, SeqScan))
				scan_method = "SeqScan";
			else if (IsA(plan, IndexScan))
				scan_method = "IndexScan";
			else if (IsA(plan, IndexOnlyScan))
				scan_method = "IndexOnlyScan";
			else if (IsA(plan, BitmapHeapScan))
				scan_method = "BitmapScan";
			else if (IsA(plan, TidScan))
				scan_method = "TidScan";
			if (scan_method != NULL)
			{
				if (methods->len > 0)
					appendStringInfoChar(methods, '\n');
				appendStringInfo(methods, "%s(%s)", scan_method, alias);
			}
		}
		*tree_out = pstrdup(alias);
		*aliases_out = pstrdup(alias);
		return true;
	}

	if (neurqo_plan_is_join(plan))
	{
		Plan* left_plan = plan->lefttree;
		Plan* right_plan = plan->righttree;
		char* left_tree;
		char* left_aliases;
		char* right_tree;
		char* right_aliases;
		const char* method;

		if (swap_hash_inputs && IsA(plan, HashJoin))
		{
			left_plan = plan->righttree;
			right_plan = plan->lefttree;
		}
		if (!neurqo_plan_hint_part(left_plan, q, methods,
								  &left_tree, &left_aliases,
								  include_methods, swap_hash_inputs) ||
			!neurqo_plan_hint_part(right_plan, q, methods,
								   &right_tree, &right_aliases,
								   include_methods, swap_hash_inputs))
			return false;
		method = IsA(plan, HashJoin) ? "HashJoin" :
			IsA(plan, MergeJoin) ? "MergeJoin" : "NestLoop";
		if (include_methods)
		{
			if (methods->len > 0)
				appendStringInfoChar(methods, '\n');
			appendStringInfo(methods, "%s(%s %s)",
							 method, left_aliases, right_aliases);
		}
		*tree_out = psprintf("(%s %s)", left_tree, right_tree);
		*aliases_out = psprintf("%s %s", left_aliases, right_aliases);
		pfree(left_tree);
		pfree(left_aliases);
		pfree(right_tree);
		pfree(right_aliases);
		return true;
	}

	if (plan->lefttree != NULL && plan->righttree == NULL)
		return neurqo_plan_hint_part(plan->lefttree, q, methods,
									tree_out, aliases_out,
									include_methods, swap_hash_inputs);
	if (plan->righttree != NULL && plan->lefttree == NULL)
		return neurqo_plan_hint_part(plan->righttree, q, methods,
									tree_out, aliases_out,
									include_methods, swap_hash_inputs);
	return false;
}

static char*
neurqo_build_plan_hint(PlannedStmt* plannedstmt, Query* q)
{
	StringInfoData methods;
	char* tree;
	char* aliases;
	char* result;

	if (plannedstmt == NULL || plannedstmt->planTree == NULL)
		return NULL;
	initStringInfo(&methods);
	if (!neurqo_plan_hint_part(plannedstmt->planTree, q, &methods,
							   &tree, &aliases, true, false))
	{
		pfree(methods.data);
		return NULL;
	}
	if (methods.len > 0)
		appendStringInfoChar(&methods, '\n');
	appendStringInfo(&methods, "Leading(%s)", tree);
	result = methods.data;
	pfree(tree);
	pfree(aliases);
	return result;
}

static char*
neurqo_build_leading_hint(PlannedStmt* plannedstmt, Query* q,
						  bool swap_hash_inputs)
{
	StringInfoData ignored_methods;
	char* tree;
	char* aliases;
	char* result;

	if (plannedstmt == NULL || plannedstmt->planTree == NULL)
		return NULL;
	initStringInfo(&ignored_methods);
	if (!neurqo_plan_hint_part(plannedstmt->planTree, q, &ignored_methods,
							   &tree, &aliases, false, swap_hash_inputs))
	{
		pfree(ignored_methods.data);
		return NULL;
	}
	result = psprintf("Leading(%s)", tree);
	pfree(ignored_methods.data);
	pfree(tree);
	pfree(aliases);
	return result;
}

static void
neurqo_plan_summary(Plan* plan, int depth, int* nnodes, int* njoins,
					int* nscans, int* max_depth)
{
	if (plan == NULL)
		return;
	(*nnodes)++;
	if (neurqo_plan_is_join(plan))
		(*njoins)++;
	if (neurqo_plan_is_scan(plan))
		(*nscans)++;
	if (depth > *max_depth)
		*max_depth = depth;
	neurqo_plan_summary(plan->lefttree, depth + 1, nnodes, njoins, nscans,
						max_depth);
	neurqo_plan_summary(plan->righttree, depth + 1, nnodes, njoins, nscans,
						max_depth);
}

static void
neurqo_append_aliases_json(Query* q, StringInfo out)
{
	ListCell* lc;
	bool first = true;

	appendStringInfoChar(out, '[');
	foreach(lc, q->rtable)
	{
		RangeTblEntry* rte = (RangeTblEntry*)lfirst(lc);

		if (rte->rtekind != RTE_RELATION)
			continue;
		if (!first)
			appendStringInfoChar(out, ',');
		neurqo_append_json_string(out, rte->eref && rte->eref->aliasname ?
								  rte->eref->aliasname : get_rel_name(rte->relid));
		first = false;
	}
	appendStringInfoChar(out, ']');
}

static void
neurqo_append_plan_json(Plan* plan, Query* q, StringInfo out, int* nnodes)
{
	Index scanrelid;

	if (plan == NULL)
	{
		appendStringInfoString(out, "null");
		return;
	}
	if (*nnodes >= NEURQO_AJA_PLAN_MAX_NODES)
	{
		appendStringInfoString(out, "{\"truncated\":true}");
		return;
	}
	(*nnodes)++;
	appendStringInfoString(out, "{");
	appendStringInfoString(out, "\"node\":");
	neurqo_append_json_string(out, neurqo_plan_node_name(plan));
	appendStringInfo(out, ",\"rows\":%.0f,\"startup_cost\":%.2f,"
					 "\"total_cost\":%.2f,\"width\":%d",
					 plan->plan_rows, plan->startup_cost,
					 plan->total_cost, plan->plan_width);
	scanrelid = neurqo_plan_scanrelid(plan);
	if (scanrelid > 0 && scanrelid <= list_length(q->rtable))
	{
		RangeTblEntry* rte = (RangeTblEntry*)list_nth(q->rtable, scanrelid - 1);

		if (rte->rtekind == RTE_RELATION)
		{
			appendStringInfoString(out, ",\"alias\":");
			neurqo_append_json_string(out, rte->eref && rte->eref->aliasname ?
									  rte->eref->aliasname :
									  get_rel_name(rte->relid));
		}
	}
	if (plan->lefttree != NULL || plan->righttree != NULL)
	{
		appendStringInfoString(out, ",\"children\":[");
		neurqo_append_plan_json(plan->lefttree, q, out, nnodes);
		if (plan->righttree != NULL)
		{
			appendStringInfoChar(out, ',');
			neurqo_append_plan_json(plan->righttree, q, out, nnodes);
		}
		appendStringInfoChar(out, ']');
	}
	appendStringInfoChar(out, '}');
}

static char*
neurqo_build_low_state(Query* q,
					   int round, PlannedStmt* selected_plan,
					   double cumulative_cost_ms,
					   int max_split_rounds,
					   bool is_split_execution)
{
	StringInfoData state;
	Plan* plan = selected_plan != NULL ? selected_plan->planTree : NULL;
	int nnodes = 0;
	int njoins = 0;
	int nscans = 0;
	int max_depth = 0;
	int plan_json_nodes = 0;

	initStringInfo(&state);
	appendStringInfo(&state,
					 "{\"request_type\":\"low\",\"pid\":%d,"
					 "\"run_id\":" UINT64_FORMAT ",\"round\":%d,"
					 "\"base_rels\":%d,\"cumulative_cost_ms\":%.3f,"
					 "\"max_split_rounds\":%d,"
					 "\"is_split_execution\":%s,\"search_strategy\":",
					 MyProcPid, neurqo_current_run_id, round,
					 list_length(q->rtable), cumulative_cost_ms,
					 max_split_rounds,
					 is_split_execution ? "true" : "false");
	neurqo_append_json_string(&state,
							  neurqo_search_enabled() ?
							  neurqo_current_search_strategy : "default");
	appendStringInfo(&state, ",\"search_k\":%d",
					 neurqo_current_search_k > 0 ?
					 neurqo_current_search_k : neurqo_search_topk);

	if (plan == NULL)
		appendStringInfoString(&state, ",\"plan_available\":false");
	else
	{
		neurqo_plan_summary(plan, 1, &nnodes, &njoins, &nscans, &max_depth);
		appendStringInfo(&state,
						 ",\"plan_available\":true,"
						 "\"plan_total_cost\":%.2f,\"plan_rows\":%.0f,"
						 "\"plan_width\":%d,"
						 "\"plan_summary\":{\"nodes\":%d,\"joins\":%d,"
						 "\"scans\":%d,\"max_depth\":%d},\"plan_json\":",
						 plan->total_cost, plan->plan_rows, plan->plan_width,
						 nnodes, njoins, nscans, max_depth);
		neurqo_append_plan_json(plan, q, &state, &plan_json_nodes);
	}
	appendStringInfoChar(&state, '}');
	return state.data;
}

static char*
neurqo_build_planner_hint(Query* q)
{
	char* search_hint = NULL;
	char* aja_hint = NULL;
	char* hint_query = NULL;

	if (neurqo_search_enabled())
		search_hint = neurqo_build_search_hint(q, NULL);
	if (neurqo_aja_enabled())
	{
		if (strcmp(neurqo_current_execution_action, "hashjoin") == 0 ||
			strcmp(neurqo_current_execution_action, "nestloop") == 0 ||
			strcmp(neurqo_current_execution_action, "mergejoin") == 0)
			aja_hint = neurqo_build_join_method_hint(
				q, neurqo_current_execution_action);
	}

	hint_query = neurqo_make_hint_query(aja_hint, search_hint);
	if (search_hint != NULL)
		pfree(search_hint);
	if (aja_hint != NULL)
		pfree(aja_hint);
	return hint_query;
}

static PlannedStmt*
neurqo_plan_execution(Query* q, const char* query_string,
					  int round, int length, int remaining,
					  double cumulative_cost_ms, int max_split_rounds,
					  bool is_split_execution,
					  double* policy_ms,
					  char** search_state_json_out,
					  char** low_state_json_out)
{
	PlannedStmt* selected_plan;
	PlannedStmt* final_plan;
	PlannedStmt* adaptive_fallback_plan = NULL;
	PlannedStmt* nestloop_plan = NULL;
	PlannedStmt* search_candidate_plan = NULL;
	char* search_hint_body = NULL;
	char* search_hint_query = NULL;
	char* aja_hint_body = NULL;
	char* lip_plan_hint_body = NULL;
	char* lip_plan_hint_query = NULL;
	char* final_hint_query = NULL;
	char* adaptive_nest_leading_body = NULL;
	char* adaptive_nest_hint_query = NULL;
	char* adaptive_baseline_leading_body = NULL;
	char* adaptive_baseline_hint_query = NULL;
	const char* search_planner_query;
	const char* execution_hint_body;
	const char* execution_planner_query;
	const char* adaptive_level;
	double search_policy_ms = 0.0;
	double low_policy_ms = 0.0;
	double lip_ms = 0.0;
	int lip_filters = 0;
	int adaptive_threshold = 0;
	int adaptive_joins = 0;
	bool adaptive_safe;
	bool forced_hash_baseline = false;

	*policy_ms = 0.0;
	neurqo_last_search_ms = 0.0;
	neurqo_last_search_candidate_cost = 0.0;
	neurqo_last_search_candidates = 0;
	neurqo_last_search_applied = false;
	neurqo_last_lip_build_ms = 0.0;
	neurqo_last_lip_filters = 0;
	neurqo_last_adaptive_joins = 0;
	neurqo_last_adaptive_threshold = 0;
	if (search_state_json_out != NULL)
		*search_state_json_out = NULL;
	if (low_state_json_out != NULL)
		*low_state_json_out = NULL;
	if (!neurqo_policy_search(q, query_string, round, length, remaining,
							  cumulative_cost_ms, max_split_rounds,
							  &search_policy_ms,
							  search_state_json_out))
	{
		neurqo_current_search_strategy[0] = '\0';
		neurqo_current_search_k = 0;
	}
	*policy_ms += search_policy_ms;

	if (neurqo_search_enabled())
		search_hint_body = neurqo_build_search_hint(
			q, &search_candidate_plan);
	search_hint_query = neurqo_make_hint_query(NULL, search_hint_body);
	search_planner_query =
		search_hint_query != NULL ? search_hint_query : query_string;
	if (search_candidate_plan != NULL)
	{
		selected_plan = search_candidate_plan;
		elog(LOG, "[neurqo] run=" UINT64_FORMAT
			 " round %d: reuse best top-k candidate plan",
			 neurqo_current_run_id, round);
	}
	else
		selected_plan = neurqo_plan_direct(copyObjectImpl(q),
										  CURSOR_OPT_PARALLEL_OK, false,
										  search_planner_query, false);

	if (!neurqo_policy_low(q, round, selected_plan,
						   cumulative_cost_ms,
						   max_split_rounds, is_split_execution,
						   &aja_hint_body,
						   &low_policy_ms, low_state_json_out))
	{
		neurqo_current_execution_action[0] = '\0';
		neurqo_current_lip_action[0] = '\0';
	}
	*policy_ms += low_policy_ms;

	adaptive_level = neurqo_adaptive_aja_level();
	adaptive_safe = adaptive_level != NULL &&
		!contain_volatile_functions((Node *) q);
	if (adaptive_level != NULL && aja_hint_body != NULL)
	{
		pfree(aja_hint_body);
		aja_hint_body = NULL;
	}
	if (aja_hint_body == NULL &&
		adaptive_level == NULL &&
		(strcmp(neurqo_current_execution_action, "hashjoin") == 0 ||
		 strcmp(neurqo_current_execution_action, "nestloop") == 0 ||
		 strcmp(neurqo_current_execution_action, "mergejoin") == 0))
		aja_hint_body = neurqo_build_join_method_hint(
			q, neurqo_current_execution_action);

	neurqo_apply_lip(q, selected_plan, &lip_ms, &lip_filters);
	neurqo_last_lip_build_ms = lip_ms;
	neurqo_last_lip_filters = lip_filters;
	execution_hint_body = search_hint_body;
	execution_planner_query = search_planner_query;
	if (lip_filters > 0)
	{
		lip_plan_hint_body = neurqo_build_plan_hint(selected_plan, q);
		lip_plan_hint_query =
			neurqo_make_hint_query(NULL, lip_plan_hint_body);
		if (lip_plan_hint_query != NULL)
		{
			execution_hint_body = lip_plan_hint_body;
			execution_planner_query = lip_plan_hint_query;
		}
	}
	if (adaptive_safe)
	{
		Query* nestloop_query = copyObjectImpl(q);
		const char* nestloop_planner_query;
		int eligible_hashjoins;
		int max_nestloop_cost_ratio_pct;

		adaptive_threshold =
			strcmp(adaptive_level, "conservative") == 0 ?
			neurqo_aja_conservative_rows : neurqo_aja_aggressive_rows;
		max_nestloop_cost_ratio_pct =
			strcmp(adaptive_level, "conservative") == 0 ?
			neurqo_aja_max_nestloop_cost_ratio_pct :
			neurqo_aja_aggressive_max_nestloop_cost_ratio_pct;
		neurqo_last_adaptive_threshold = adaptive_threshold;
		/*
		 * Workload SQL can carry physical-method hints from offline
		 * generation. Preserve the selected logical order, but let PostgreSQL
		 * choose the baseline physical methods before building AJA branches.
		 */
		adaptive_baseline_leading_body =
			neurqo_build_leading_hint(selected_plan, q, false);
		adaptive_baseline_hint_query =
			neurqo_make_hint_query(NULL, adaptive_baseline_leading_body);
		if (lip_filters == 0)
			adaptive_fallback_plan = selected_plan;
		else
			adaptive_fallback_plan = neurqo_plan_direct(
				copyObjectImpl(q), CURSOR_OPT_PARALLEL_OK, false,
				execution_planner_query != NULL ?
					execution_planner_query : query_string, true);
		if (adaptive_baseline_hint_query != NULL)
			final_plan = neurqo_plan_direct(
				copyObjectImpl(q), CURSOR_OPT_PARALLEL_OK, false,
				adaptive_baseline_hint_query, true);
		else
			final_plan = adaptive_fallback_plan;
		eligible_hashjoins =
			neurqo_count_adaptive_hashjoins(final_plan);
		/*
		 * Offline workload SQL often pins every join to NestLoop. If the
		 * same logical order has no natural HashJoin, form a hash-oriented
		 * baseline so the executor can compare it with the NestLoop plan.
		 */
		if (eligible_hashjoins == 0)
		{
			PlannedStmt* hashjoin_plan =
				neurqo_plan_hashjoin_candidate(
					copyObjectImpl(q), adaptive_baseline_hint_query);
			int forced_eligible =
				neurqo_count_adaptive_hashjoins(hashjoin_plan);

			if (forced_eligible > 0)
			{
				final_plan = hashjoin_plan;
				eligible_hashjoins = forced_eligible;
				forced_hash_baseline = true;
			}
		}
		if (eligible_hashjoins > 0)
		{
			adaptive_nest_leading_body =
				neurqo_build_leading_hint(final_plan, nestloop_query, true);
			adaptive_nest_hint_query =
				neurqo_make_hint_query(NULL, adaptive_nest_leading_body);
			nestloop_planner_query = adaptive_nest_hint_query != NULL ?
				adaptive_nest_hint_query : execution_planner_query;
			nestloop_plan =
				neurqo_plan_nestloop_candidate(nestloop_query,
											  nestloop_planner_query);
			if (nestloop_plan != NULL)
			{
				adaptive_joins =
					neurqo_wrap_adaptive_joins(final_plan, nestloop_plan,
											  adaptive_level,
											  adaptive_threshold,
											  max_nestloop_cost_ratio_pct,
											  neurqo_current_run_id, round);
				neurqo_last_adaptive_joins = adaptive_joins;
			}
		}
		if (adaptive_joins == 0)
			final_plan = adaptive_fallback_plan;
		elog(LOG, "[neurqo] run=" UINT64_FORMAT
			 " round %d: adaptive join planning level=%s threshold_rows=%d "
			 "hash_baseline=%s eligible=%d wrapped=%d fallback=%s",
			 neurqo_current_run_id, round, adaptive_level,
			 adaptive_threshold,
			 forced_hash_baseline ? "forced" : "natural",
			 eligible_hashjoins, adaptive_joins,
			 adaptive_joins > 0 ? "none" :
			 eligible_hashjoins > 0 ? "no_compatible_nestloop" :
			 "no_eligible_hashjoin");
	}
	else
	{
		if (lip_filters == 0 && aja_hint_body == NULL)
			final_plan = selected_plan;
		else
		{
			final_hint_query =
				neurqo_make_hint_query(aja_hint_body, execution_hint_body);
			final_plan = neurqo_plan_direct(q, CURSOR_OPT_PARALLEL_OK, false,
										   final_hint_query != NULL ?
										   final_hint_query : query_string, true);
		}
		if (adaptive_level != NULL)
			elog(LOG, "[neurqo] run=" UINT64_FORMAT
				 " round %d: adaptive join fallback=volatile_query",
				 neurqo_current_run_id, round);
	}

	elog(LOG, "[neurqo] run=" UINT64_FORMAT
		 " round %d: execution actions search=%s k=%d execution=%s "
		 "lip=%s filters=%d adaptive_joins=%d policy_ms=%.2f",
		 neurqo_current_run_id, round,
		 neurqo_search_enabled() ? neurqo_current_search_strategy : "default",
		 neurqo_current_search_k > 0 ?
		 neurqo_current_search_k : neurqo_search_topk,
		 neurqo_aja_enabled() ? neurqo_current_execution_action : "none",
		 neurqo_lip_enabled() ? neurqo_current_lip_action : "none",
		 lip_filters, adaptive_joins, *policy_ms);

	if (search_hint_body != NULL)
		pfree(search_hint_body);
	if (search_hint_query != NULL)
		pfree(search_hint_query);
	if (aja_hint_body != NULL)
		pfree(aja_hint_body);
	if (lip_plan_hint_body != NULL)
		pfree(lip_plan_hint_body);
	if (lip_plan_hint_query != NULL)
		pfree(lip_plan_hint_query);
	if (final_hint_query != NULL)
		pfree(final_hint_query);
	if (adaptive_nest_leading_body != NULL)
		pfree(adaptive_nest_leading_body);
	if (adaptive_nest_hint_query != NULL)
		pfree(adaptive_nest_hint_query);
	return final_plan;
}

//The interface
void doQSparse(const char* query_string, CommandTag commandTag, Node* pstmt, Query* querytree, QueryCompletion* completionTag)
{
	neurqo_current_run_id = ++neurqo_run_seq;
	neurqo_reset_execution_actions();
	neurqo_current_high_action[0] = '\0';
	elog(LOG, "[neurqo] run=" UINT64_FORMAT " enter: enabled=%d cmd=%d rtable=%d alg=%d order_decision=%s sql=%s",
		 neurqo_current_run_id, neurqo_enabled ? 1 : 0, querytree->commandType,
		 list_length(querytree->rtable), query_splitting_algorithm,
		 neurqo_order_decision_name(order_decision), query_string);
	if (querytree->commandType != CMD_UTILITY && query_splitting_algorithm != Minsubquery)
	{
		//remove Redundant Join
		rRj(querytree);
	}
	elog(LOG, "[neurqo] run=" UINT64_FORMAT " after rRj: rtable=%d",
		 neurqo_current_run_id, list_length(querytree->rtable));
	PlannedStmt* plannedstmt = NULL;
	if (querytree->commandType == CMD_UTILITY)
	{
		MemoryContext oldcontext = MemoryContextSwitchTo(MessageContext);
		/* Utility commands require no planning. */
		plannedstmt = makeNode(PlannedStmt);
		plannedstmt->commandType = CMD_UTILITY;
		plannedstmt->canSetTag = querytree->canSetTag;
		plannedstmt->utilityStmt = querytree->utilityStmt;
		plannedstmt->stmt_location = querytree->stmt_location;
		plannedstmt->stmt_len = querytree->stmt_len;
		QSExecutor(query_string, commandTag, pstmt, plannedstmt, DestRemote, NULL, completionTag, querytree, NULL, NIL, oldcontext);
		return;
	}
	ListCell* lc;
	int length = 0;
	foreach(lc, querytree->rtable)
	{
		RangeTblEntry* rte = (RangeTblEntry*)lfirst(lc);

		if (rte->rtekind == RTE_JOIN)
			continue;
		if (rte->rtekind != RTE_RELATION)
		{
			MemoryContext oldcontext = MemoryContextSwitchTo(MessageContext);

			plannedstmt = neurqo_plan(querytree, CURSOR_OPT_PARALLEL_OK, true);
			QSExecutor(query_string, commandTag, pstmt, plannedstmt, DestRemote,
					   NULL, completionTag, querytree, NULL, NIL, oldcontext);
			return;
		}
		if (rte->relkind != RELKIND_RELATION)
		{
			MemoryContext oldcontext = MemoryContextSwitchTo(MessageContext);

			plannedstmt = neurqo_plan(querytree, CURSOR_OPT_PARALLEL_OK, true);
			QSExecutor(query_string, commandTag, pstmt, plannedstmt, DestRemote,
					   NULL, completionTag, querytree, NULL, NIL, oldcontext);
			return;
		}
		length++;
	}
	if (length <= 2)
	{
		MemoryContext oldcontext = MemoryContextSwitchTo(MessageContext);
		bool stop_now = true;
		double high_policy_ms = 0.0;
		double terminal_policy_ms = 0.0;
		double planning_ms;
		double high_state_ms;
		double execution_ms;
		double round_start = neurqo_now_ms();
		double t0;
		char* state_json = NULL;
		char* search_state_json = NULL;
		char* low_state_json = NULL;
		bool high_ok;

		t0 = neurqo_now_ms();
		high_ok = neurqo_policy_high(querytree, query_string, 0, length, 0,
									0.0, 1, &stop_now,
									&high_policy_ms, &state_json);
		high_state_ms = Max(
			neurqo_now_ms() - t0 - high_policy_ms, 0.0);
		if (!stop_now)
			elog(LOG, "[neurqo] run=" UINT64_FORMAT
				 " high requested split for an unsplittable query; forcing stop",
				 neurqo_current_run_id);
		t0 = neurqo_now_ms();
		if (high_ok)
			plannedstmt = neurqo_plan_execution(
				querytree, query_string, 0, length, 0, 0.0, 1,
				false,
				&terminal_policy_ms, &search_state_json,
				&low_state_json);
		else
			plannedstmt = neurqo_plan(querytree, CURSOR_OPT_PARALLEL_OK, false);
		planning_ms = high_state_ms + neurqo_now_ms() - t0;
		t0 = neurqo_now_ms();
		QSExecutor(query_string, commandTag, pstmt, plannedstmt, DestRemote, NULL, completionTag, querytree, NULL, NIL, oldcontext);
		execution_ms = neurqo_now_ms() - t0;
			neurqo_log_trajectory_event(
				"final", 0, state_json, true, NULL, search_state_json,
				low_state_json, plannedstmt, querytree,
				high_policy_ms + terminal_policy_ms,
				planning_ms, execution_ms, neurqo_now_ms() - round_start,
				"remote");
		if (state_json != NULL)
			pfree(state_json);
		if (search_state_json != NULL)
			pfree(search_state_json);
		if (low_state_json != NULL)
			pfree(low_state_json);
		return;
	}
	//split parent query by foreign key
	Recon(query_string, commandTag, pstmt, querytree, completionTag);

	return;
}

/*
 * Classify entity and relationship relations for the split graph.
 *
 * The original QuerySplit implementation also deleted predicates joining two
 * relationship relations here.  Those predicates are not generally redundant
 * (TPC-H Q5's customer-to-supplier nation predicate is one counterexample), so
 * mutating the user Query changed both results and runtime even when High chose
 * stop.  List2Graph() builds a separate acyclic scheduling graph, so keep every
 * SQL predicate intact at entry.  Only after High selects split may the
 * split-execution copy remove an R-R equality proven redundant by transitivity.
 */
static void rRj(Query* querytree)
{
	//get all the foreign key
	List* FKlist = grFK(querytree->rtable);
	int length = querytree->rtable->length;
	is_relationship = (bool*)palloc(length * sizeof(bool));
	memset(is_relationship, true, length * sizeof(bool));
	ListCell* lc;
	//referenced relation is entity
	foreach(lc, FKlist)
	{
		ForeignKeyOptInfo* fkOptInfo = (ForeignKeyOptInfo*)lfirst(lc);
		int x = fkOptInfo->ref_relid - 1;
		is_relationship[x] = false;
	}
}

static void Recon(const char* query_string, CommandTag commandTag, Node* pstmt, Query* ori_query, QueryCompletion* completionTag)
{
	MemoryContext oldcontext = MemoryContextSwitchTo(MessageContext);
	Query* global_query = copyObjectImpl(ori_query);
	PlannedStmt* plannedstmt = NULL;
	if (global_query->commandType == CMD_UTILITY)
	{
		plannedstmt = neurqo_plan(
			global_query, CURSOR_OPT_PARALLEL_OK, false);
		QSExecutor(query_string, commandTag, pstmt, plannedstmt, DestRemote, NULL, completionTag, NULL, NULL, NIL, oldcontext);
		return;
	}
	int length = global_query->rtable->length;
	if (length == 1)
	{
		plannedstmt = neurqo_plan(
			global_query, CURSOR_OPT_PARALLEL_OK, false);
		QSExecutor(query_string, commandTag, pstmt, plannedstmt, DestRemote, NULL, completionTag, NULL, NULL, NIL, oldcontext);
		return;
	}
	List* RClist = NIL;
	List* Joinlist = NIL;
	List* WhereClause = NIL;
	if (global_query->jointree->quals != NULL)
	{
		if (IsA(global_query->jointree->quals, BoolExpr) &&
			((BoolExpr*)global_query->jointree->quals)->boolop == AND_EXPR)
		{
			WhereClause =
				((BoolExpr*)global_query->jointree->quals)->args;
		}
		else
			WhereClause =
				list_make1(global_query->jointree->quals);
	}
	ListCell* lc;
	foreach(lc, WhereClause)
	{
		if (is_RC(lfirst(lc)))
			RClist = lappend(RClist, lfirst(lc));
		else
			Joinlist = lappend(Joinlist, lfirst(lc));
	}
	List* FKlist = grFK(global_query->rtable);
	//transfer join list to join graph
	bool* graph = List2Graph(is_relationship, Joinlist, FKlist, length);
	//value start from 1, index start from 0
	transfer_array = (Index*)palloc(length * sizeof(Index));
	int round = 0;
	bool policy_available = true;
	double cumulative_cost_ms = 0.0;
	int max_split_rounds = Max(hasNext(graph, length), 1);
	while (true)
	{
		bool stop_now = false;
		double policy_ms = 0.0;
		double optimize_ms = 0.0;
		double exec_ms = 0.0;
		double selection_policy_ms = 0.0;
		double high_state_ms = 0.0;
		double round_start = neurqo_now_ms();
		double t0;
		int remaining = hasNext(graph, length);
		int high_remaining = remaining > 1 ? remaining : 0;
		char* state_json = NULL;
		char* selection_state_json = NULL;
		char* search_state_json = NULL;
		char* low_state_json = NULL;
		Query* selected_query;

		if (round >= neurqo_max_rounds)
			high_remaining = 0;
		if (policy_available)
		{
			t0 = neurqo_now_ms();
			policy_available = neurqo_policy_high(
				global_query, query_string, round, length, high_remaining,
				cumulative_cost_ms, max_split_rounds,
				&stop_now, &policy_ms, &state_json);
			high_state_ms = Max(
				neurqo_now_ms() - t0 - policy_ms, 0.0);
		}
		if (!policy_available || remaining <= 1)
			stop_now = true;
		if (round >= neurqo_max_rounds)
		{
			stop_now = true;
			elog(LOG, "[neurqo] run=" UINT64_FORMAT " round %d: reached neurqo.max_rounds=%d; finishing residual query",
				 neurqo_current_run_id, round, neurqo_max_rounds);
		}
		if (stop_now)
		{
			double terminal_policy_ms = 0.0;

			t0 = neurqo_now_ms();
			if (policy_available)
				plannedstmt = neurqo_plan_execution(
					global_query, query_string, round, length, remaining,
					cumulative_cost_ms, max_split_rounds,
					false,
					&terminal_policy_ms, &search_state_json,
					&low_state_json);
			else
				plannedstmt = neurqo_plan(
					global_query, CURSOR_OPT_PARALLEL_OK, false);
			policy_ms += terminal_policy_ms;
			optimize_ms = high_state_ms + neurqo_now_ms() - t0;
			t0 = neurqo_now_ms();
			QSExecutor(query_string, commandTag, pstmt, plannedstmt, DestRemote,
					   NULL, completionTag, global_query, transfer_array, FKlist,
					   oldcontext);
			exec_ms = neurqo_now_ms() - t0;
			cumulative_cost_ms += exec_ms;
			neurqo_log_trajectory_event("final", round, state_json, stop_now,
										NULL,
										search_state_json, low_state_json,
										plannedstmt, global_query,
										policy_ms, optimize_ms, exec_ms,
										neurqo_now_ms() - round_start,
										"remote");
			elog(LOG, "[neurqo] run=" UINT64_FORMAT " round %d: final residual executed policy_ms=%.2f planning_ms=%.2f execution_ms=%.2f total_ms=%.2f",
				 neurqo_current_run_id, round, policy_ms, optimize_ms, exec_ms,
				 neurqo_now_ms() - round_start);
			if (state_json != NULL)
				pfree(state_json);
			if (search_state_json != NULL)
				pfree(search_state_json);
			if (low_state_json != NULL)
				pfree(low_state_json);
			break;
		}

		/*
		 * A split decision only chooses the next execution object.  Search
		 * and Low are requested after SSA has selected that object.
		 */
		{
			int removed_equalities =
				neurqo_remove_redundant_rr_equalities(
					global_query, is_relationship, length);

			if (removed_equalities > 0)
				elog(LOG, "[neurqo] run=" UINT64_FORMAT
					 " round %d: removed %d provably redundant R-R "
					 "equalities from split execution query",
					 neurqo_current_run_id, round, removed_equalities);
		}
		t0 = neurqo_now_ms();
		selected_query = QSSelectSubquery(
			global_query, graph, transfer_array, length, query_string,
			round, cumulative_cost_ms, max_split_rounds,
			&selection_policy_ms, &selection_state_json);
		policy_ms += selection_policy_ms;
		if (selected_query == NULL)
		{
			double fallback_policy_ms = 0.0;

			elog(WARNING, "[neurqo] run=" UINT64_FORMAT
				 " round %d: no executable split candidate; finishing residual",
				 neurqo_current_run_id, round);
			plannedstmt = neurqo_plan_execution(
				global_query, query_string, round, length, remaining,
				cumulative_cost_ms, max_split_rounds,
				false,
				&fallback_policy_ms, &search_state_json,
				&low_state_json);
			policy_ms += fallback_policy_ms;
			optimize_ms = high_state_ms + neurqo_now_ms() - t0;
			t0 = neurqo_now_ms();
			QSExecutor(query_string, commandTag, pstmt, plannedstmt,
					   DestRemote, NULL, completionTag, global_query,
					   transfer_array, FKlist, oldcontext);
			exec_ms = neurqo_now_ms() - t0;
			neurqo_log_trajectory_event(
				"final", round, state_json, true, selection_state_json,
				search_state_json, low_state_json, plannedstmt, global_query,
				policy_ms, optimize_ms,
				exec_ms, neurqo_now_ms() - round_start, "remote");
			if (state_json != NULL)
				pfree(state_json);
			if (selection_state_json != NULL)
				pfree(selection_state_json);
			if (search_state_json != NULL)
				pfree(search_state_json);
			if (low_state_json != NULL)
				pfree(low_state_json);
			break;
		}
		else
		{
			double execution_policy_ms = 0.0;

			plannedstmt = neurqo_plan_execution(
				selected_query, query_string, round,
				list_length(selected_query->rtable), 0,
				cumulative_cost_ms, max_split_rounds,
				true,
				&execution_policy_ms, &search_state_json,
				&low_state_json);
			policy_ms += execution_policy_ms;
		}
		optimize_ms = high_state_ms + neurqo_now_ms() - t0;
		if (plannedstmt == NULL)
			ereport(ERROR,
					(errmsg("NeurQO could not plan selected split candidate")));
		queryId++;
		char* relname = palloc(7 * sizeof(char));
		sprintf(relname, "temp%d", queryId);
		//Execute the subquery and do some change for next subquery creation
		t0 = neurqo_now_ms();
		FKlist = QSExecutor(query_string, commandTag, pstmt, plannedstmt,
						   DestIntoRel, relname, completionTag, global_query,
						   transfer_array, FKlist, oldcontext);
		exec_ms = neurqo_now_ms() - t0;
		cumulative_cost_ms += exec_ms;
		neurqo_log_trajectory_event("split", round, state_json, false,
									selection_state_json,
									search_state_json, low_state_json,
									plannedstmt, selected_query, policy_ms,
									optimize_ms, exec_ms,
									neurqo_now_ms() - round_start,
									relname);
		elog(LOG, "[neurqo] run=" UINT64_FORMAT " round %d: apply split result=%s policy_ms=%.2f split_planning_ms=%.2f execution_rewrite_ms=%.2f total_ms=%.2f",
			 neurqo_current_run_id, round,
			 relname,
			 policy_ms, optimize_ms, exec_ms, neurqo_now_ms() - round_start);
		if (state_json != NULL)
			pfree(state_json);
		if (selection_state_json != NULL)
			pfree(selection_state_json);
		if (search_state_json != NULL)
			pfree(search_state_json);
		if (low_state_json != NULL)
			pfree(low_state_json);
		WhereClause = NIL;
		if (global_query->jointree->quals != NULL)
		{
			if (IsA(global_query->jointree->quals, BoolExpr) &&
				((BoolExpr*)global_query->jointree->quals)->boolop ==
				AND_EXPR)
			{
				WhereClause =
					((BoolExpr*)global_query->jointree->quals)->args;
			}
			else
				WhereClause =
					list_make1(global_query->jointree->quals);
		}
		Joinlist = NIL;
		ListCell* lc;
		foreach(lc, WhereClause)
		{
			if (!is_RC(lfirst(lc)))
				Joinlist = lappend(Joinlist, lfirst(lc));
		}
		length = global_query->rtable->length;
		graph = List2Graph(is_relationship, Joinlist, FKlist, length);
		round++;
	}
	pfree(transfer_array);
	pfree(graph);
	transfer_array = NULL;
	graph = NULL;
	return;
}

typedef struct NeurqoCandidateValidationContext
{
	int			nrels;
	bool		valid;
} NeurqoCandidateValidationContext;

static bool
neurqo_candidate_reference_walker(
	Node* node, NeurqoCandidateValidationContext* context)
{
	if (node == NULL || !context->valid)
		return false;
	if (IsA(node, Var))
	{
		Var* var = (Var*)node;

		if (var->varlevelsup == 0 &&
			(var->varno < 1 || var->varno > context->nrels))
			context->valid = false;
		return false;
	}
	if (IsA(node, RangeTblRef))
	{
		RangeTblRef* ref = (RangeTblRef*)node;

		if (ref->rtindex < 1 || ref->rtindex > context->nrels)
			context->valid = false;
		return false;
	}
	if (IsA(node, Query))
		return query_tree_walker(
			(Query*)node,
			(bool (*)())neurqo_candidate_reference_walker,
			context, QTW_IGNORE_RT_SUBQUERIES);
	return expression_tree_walker(
		node, (bool (*)())neurqo_candidate_reference_walker, context);
}

static bool
neurqo_split_candidate_valid(Query* query, int center_x, int center_y)
{
	NeurqoCandidateValidationContext context;

	context.nrels = list_length(query->rtable);
	context.valid = true;
	(void)neurqo_candidate_reference_walker((Node*)query, &context);
	if (!context.valid)
		ereport(WARNING,
				(errmsg("NeurQO skipped an invalid split candidate"),
				 errdetail("center=(%d,%d) references an absent range table",
						   center_x, center_y)));
	return context.valid;
}

//Generate the current QSA candidates, then ask SSA which one to execute.
static Query*
QSSelectSubquery(Query* global_query, bool* graph, Index* transfer_array,
				 int length, const char* query_string, int round,
				 double cumulative_cost_ms, int max_split_rounds,
				 double* policy_ms, char** selection_state_json_out)
{
	List* candidates = NIL;
	ListCell* lc;
	NeurqoSplitCandidate* selected = NULL;
	NeurqoSplitCandidate* fallback = NULL;
	PlannedStmt* fallback_plan = NULL;
	Index rels[2] = {0, 0};
	int selected_id = -1;
	int X;
	int Y;

	*policy_ms = 0.0;
	if (selection_state_json_out != NULL)
		*selection_state_json_out = NULL;
	if (graph == NULL || length <= 1 || hasNext(graph, length) <= 1)
		return NULL;

	/*
	 * A split action always executes an intermediate object.  The final
	 * residual is handled by the stop branch and is never selected here.
	 */
	mydest = DestIntoRel;

	if (order_decision == global_view)
	{
		PlannedStmt* temp = neurqo_plan(
			copyObjectImpl(global_query), CURSOR_OPT_PARALLEL_OK, false);
		int leaf_has = 0;
		int depth = 0;

		if (temp != NULL && temp->planTree != NULL)
		{
			Plan* temp_plan = find_node_with_nleaf_recursive(
				temp->planTree, 2, &leaf_has, &depth);

			if (temp_plan != NULL)
			{
				walk_plantree(temp_plan, rels);
				if (rels[0] > 0 && rels[0] <= length &&
					rels[1] > 0 && rels[1] <= length)
				{
					rels[0] = ((RangeTblEntry*)list_nth(
						global_query->rtable, rels[0] - 1))->relid;
					rels[1] = ((RangeTblEntry*)list_nth(
						global_query->rtable, rels[1] - 1))->relid;
				}
			}
		}
	}

	if (query_splitting_algorithm == RelationshipCenter ||
		query_splitting_algorithm == EntityCenter)
	{
		for (int i = 0; i < length; i++)
		{
			List* rtable;
			Query* local_query;
			NeurqoSplitCandidate* candidate;

			for (int j = 0; j < length; j++)
				transfer_array[j] = 0;
			rtable = getRT_2(
				global_query->rtable, graph, length, i, transfer_array);
			if (list_length(rtable) < 2)
				continue;

			local_query = createQuery(
				global_query, DestIntoRel, rtable, transfer_array, length);
			candidate = palloc0(sizeof(NeurqoSplitCandidate));
			candidate->x = i;
			candidate->y = -1;
			candidate->query = local_query;
			if (!neurqo_split_candidate_valid(
					local_query, candidate->x, candidate->y))
			{
				pfree(candidate);
				continue;
			}
			candidate->estimate_plan = neurqo_plan(
				copyObjectImpl(local_query), CURSOR_OPT_PARALLEL_OK, false);
			candidate->candidate_id = list_length(candidates);
			candidates = lappend(candidates, candidate);
		}
	}
	else if (query_splitting_algorithm == Minsubquery)
	{
		for (int i = 0; i < length; i++)
		{
			for (int j = i + 1; j < length; j++)
			{
				List* rtable;
				Query* local_query;
				NeurqoSplitCandidate* candidate;

				for (int k = 0; k < length; k++)
					transfer_array[k] = 0;
				rtable = getRT_1(
					global_query->rtable, graph, length, i, j,
					transfer_array);
				if (rtable == NIL)
					continue;

				local_query = createQuery(
					global_query, DestIntoRel, rtable, transfer_array,
					length);
				candidate = palloc0(sizeof(NeurqoSplitCandidate));
				candidate->x = i;
				candidate->y = j;
				candidate->query = local_query;
				if (!neurqo_split_candidate_valid(
						local_query, candidate->x, candidate->y))
				{
					pfree(candidate);
					continue;
				}
				candidate->estimate_plan = neurqo_plan(
					copyObjectImpl(local_query), CURSOR_OPT_PARALLEL_OK,
					false);
				candidate->candidate_id = list_length(candidates);
				candidates = lappend(candidates, candidate);
			}
		}
	}

	if (candidates == NIL)
		return NULL;

	/* Preserve QuerySplit's configured SSA as the local failure fallback. */
	foreach(lc, candidates)
	{
		NeurqoSplitCandidate* candidate =
			(NeurqoSplitCandidate*)lfirst(lc);

		if (candidate->estimate_plan == NULL)
			continue;
		if (fallback_plan == NULL ||
			tarfunc(rels, candidate->estimate_plan, fallback_plan) == NEWBETTER)
		{
			fallback = candidate;
			fallback_plan = candidate->estimate_plan;
		}
	}
	if (fallback == NULL)
		fallback = (NeurqoSplitCandidate*)linitial(candidates);

	if (neurqo_policy_select(
			global_query, query_string, round, candidates,
			cumulative_cost_ms, max_split_rounds, &selected_id,
			policy_ms, selection_state_json_out))
		selected = (NeurqoSplitCandidate*)list_nth(candidates, selected_id);
	else
	{
		selected = fallback;
		neurqo_current_candidate_id = selected->candidate_id;
		snprintf(neurqo_current_selection_strategy,
				 sizeof(neurqo_current_selection_strategy),
				 "querysplit-fallback");
		elog(LOG, "[neurqo] run=" UINT64_FORMAT
			 " round %d: select fallback candidate_id=%d strategy=%s",
			 neurqo_current_run_id, round, selected->candidate_id,
			 neurqo_order_decision_name(order_decision));
	}

	X = selected->x;
	Y = selected->y;
	for (int j = 0; j < length; j++)
		transfer_array[j] = 0;

	if (query_splitting_algorithm == RelationshipCenter ||
		query_splitting_algorithm == EntityCenter)
	{
		Index index = 1;

		for (int j = 0; j < length; j++)
		{
			if (graph[X * length + j] || X == j)
				transfer_array[j] = index++;
		}
		for (int j = 0; j < length; j++)
		{
			if (graph[X * length + j])
			{
				graph[X * length + j] = false;
				graph[j * length + X] = false;
			}
		}
	}
	else
	{
		transfer_array[X] = 1;
		transfer_array[Y] = 2;
		graph[X * length + Y] = false;
		for (int i = 0; i < length; i++)
		{
			if (i < X &&
				graph[i * length + X] && graph[i * length + Y])
				graph[i * length + X] = false;
			else if (i > X && i < Y &&
					 graph[X * length + i] && graph[i * length + Y])
				graph[X * length + i] = false;
			else if (i > Y &&
					 graph[X * length + i] && graph[Y * length + i])
				graph[X * length + i] = false;
		}
	}

	if (global_query->jointree->quals != NULL)
	{
		if (IsA(global_query->jointree->quals, BoolExpr) &&
			((BoolExpr*)global_query->jointree->quals)->boolop == AND_EXPR)
		{
			BoolExpr* quals = (BoolExpr*)global_query->jointree->quals;

			quals->args = simplifyjoinlist(
				quals->args, DestIntoRel, transfer_array, graph, length);
			if (quals->args == NIL)
				global_query->jointree->quals = NULL;
		}
		else
		{
			List* filtered = simplifyjoinlist(
				list_make1(global_query->jointree->quals),
				DestIntoRel, transfer_array, graph, length);

			global_query->jointree->quals =
				filtered == NIL ? NULL : (Node*)linitial(filtered);
		}
	}
	return selected->query;
}

//Executor
static List* QSExecutor(const char* query_string, CommandTag commandTag, Node* pstmt, PlannedStmt* plannedstmt, CommandDest dest, char* relname, QueryCompletion* completionTag, Query* querytree, Index* transfer_array, List* FKlist, MemoryContext oldcontext)
{
	Oid relid;
	int16 format;
	Portal portal;
	List* plantree_list;
	DestReceiver* receiver = NULL;
	bool is_parallel_worker = false;
	double t0;

	neurqo_reset_execution_metrics();
	BeginCommand(commandTag, dest);
	plantree_list = lappend(NIL, plannedstmt);
	CHECK_FOR_INTERRUPTS();
	portal = CreatePortal("", true, true);
	portal->visible = false;
	PortalDefineQuery(portal, NULL, query_string, commandTag, plantree_list, NULL);
	PortalStart(portal, NULL, 0, InvalidSnapshot);
	format = 0;
	PortalSetResultFormat(portal, 1, &format);
	if (dest == DestRemote)
	{
		receiver = CreateDestReceiver(dest);
		SetRemoteDestReceiverParams(receiver, portal);
	}
	if (dest == DestIntoRel)
	{
		IntoClause* into = makeNode(IntoClause);
		into->rel = makeRangeVar(NULL, relname, plannedstmt->stmt_location);
		into->rel->relpersistence = RELPERSISTENCE_TEMP;
		into->onCommit = ONCOMMIT_NOOP;
		//into->onCommit = ONCOMMIT_DROP;
		into->rel->inh = false;
		into->skipData = false;
		into->viewQuery = NULL;
		receiver = CreateIntoRelDestReceiver(into);
	}
	MemoryContextSwitchTo(oldcontext);
	//Executor
	t0 = neurqo_now_ms();
	(void)PortalRun(portal, FETCH_ALL, true, true, receiver, receiver, completionTag);
	neurqo_last_executor_ms = neurqo_now_ms() - t0;
	if (dest == DestIntoRel)
	{
		RangeVar* temp_relation = ((DR_intorel*)receiver)->into->rel;

		CommandCounterIncrement();
		relid = RangeVarGetRelid(temp_relation, NoLock, false);
		neurqo_last_materialized_rows = completionTag->nprocessed;
		neurqo_last_materialized_bytes = neurqo_total_relation_size(relid);
		t0 = neurqo_now_ms();
		neurqo_analyze_temp_relation(relid, temp_relation);
		neurqo_last_analyze_ms = neurqo_now_ms() - t0;
		CommandCounterIncrement();
		t0 = neurqo_now_ms();
		FKlist = Prepare4Next(querytree, transfer_array, (DR_intorel*)receiver, plannedstmt, relname, FKlist);
		neurqo_last_residual_rewrite_ms = neurqo_now_ms() - t0;
		CommandCounterIncrement();
	}
	receiver->rDestroy(receiver);
	PortalDrop(portal, false);
	EndCommand(completionTag, dest, false);
	return FKlist;
}

static List* Prepare4Next(Query* global_query, Index* transfer_array, DR_intorel* receiver, PlannedStmt* plannedstmt, char* relname, List* FKlist)
{
	int length = global_query->rtable->length;
	int X = -1;
	int before = 0;
	for (int i = 0; i < length; i++)
	{
		if (X == -1 && transfer_array[i] == 0)
		{
			continue;
		}
		else if (X == -1 && transfer_array[i] != 0)
		{
			X = i;
			before = 0;
			if (query_splitting_algorithm == RelationshipCenter)
				is_relationship[i - before] = false;
			else if (query_splitting_algorithm == EntityCenter)
				is_relationship[i - before] = true;
		}
		else if (transfer_array[i] != 0)
		{
			before++;
		}
		else if (transfer_array[i] == 0)
		{
			if (query_splitting_algorithm == RelationshipCenter || query_splitting_algorithm == EntityCenter)
				is_relationship[i - before] = is_relationship[i];
		}
	}
	ListCell* lc;
	ListCell* prev = NULL;
	foreach(lc, FKlist)
	{
		bool flag = false;
		ForeignKeyOptInfo* fkOptInfo = (ForeignKeyOptInfo*)lfirst(lc);
		int x = fkOptInfo->con_relid - 1;
		int y = fkOptInfo->ref_relid - 1;
		if (transfer_array[x] != 0 && transfer_array[y] != 0)
		{
			FKlist = foreach_delete_current(FKlist, lc);
		}
		else if (transfer_array[x] != 0)
		{
			fkOptInfo->con_relid = X + 1;
			prev = lc;
		}
		else if (transfer_array[y] != 0)
		{
			fkOptInfo->ref_relid = X + 1;
			prev = lc;
		}
		else
		{
			prev = lc;
		}
	}

	Oid relid = RangeVarGetRelid(receiver->into->rel, NoLock, true);
	Relation relation = table_open(relid, NoLock);
	List* varlist = pull_var_clause((Node*)global_query->jointree, 0);
	foreach(lc, varlist)
	{
		Var* var = (Var*)lfirst(lc);
		Index source_varno = neurqo_source_varno(var, length);
		AttrNumber source_attno = neurqo_source_attno(var);

		if (source_varno == 0 || source_attno <= 0)
			ereport(ERROR,
					(errmsg("NeurQO cannot rewrite an invalid residual Var"),
					 errdetail("varno=%u varnosyn=%u varattno=%d varattnosyn=%d",
							   var->varno, var->varnosyn,
							   var->varattno, var->varattnosyn)));
		if (transfer_array[source_varno - 1] != 0)
		{
			RangeTblEntry* rte =
				(RangeTblEntry*)list_nth(global_query->rtable,
										source_varno - 1);
			int len;
			char* attrname;
			bool found = false;

			if (rte->eref == NULL ||
				source_attno > list_length(rte->eref->colnames))
				ereport(ERROR,
						(errmsg("NeurQO cannot resolve a residual column"),
						 errdetail("relation=%u attribute=%d",
								   source_varno, source_attno)));
			len = strlen(rte->eref->aliasname) +
				strlen(strVal(list_nth(rte->eref->colnames,
									  source_attno - 1))) + 2;
			attrname = (char*)palloc(len * sizeof(char));
			sprintf(attrname, "%s_%s", rte->eref->aliasname,
					strVal(list_nth(rte->eref->colnames,
									source_attno - 1)));
			var->varno = X + 1;
			var->varnosyn = var->varno;
			for (int i = 0; i < relation->rd_att->natts; i++)
			{
				if (strcmp(attrname, relation->rd_att->attrs[i].attname.data) == 0)
				{
					Form_pg_attribute attr =
						TupleDescAttr(relation->rd_att, i);

					found = true;
					var->varattno = i + 1;
					var->varattnosyn = var->varattno;
					var->vartype = attr->atttypid;
					var->vartypmod = attr->atttypmod;
					var->varcollid = attr->attcollation;
					break;
				}
			}
			if (!found)
				ereport(ERROR,
						(errmsg("NeurQO materialized a split without a required residual column"),
						 errdetail("temporary relation=%s missing column=%s",
								   relname, attrname)));
			pfree(attrname);
			attrname = NULL;
		}
		else
		{
			int before = 0;
			for (int i = X + 1; i < var->varno - 1; i++)
			{
				if (transfer_array[i] != 0)
				{
					before++;
				}
			}
			var->varno -= before;
			var->varnosyn = var->varno;
		}
	}
	foreach(lc, FKlist)
	{
		ForeignKeyOptInfo* fkOptInfo = (ForeignKeyOptInfo*)lfirst(lc);
		int before = 0;
		for (int i = X + 1; i < fkOptInfo->con_relid - 1; i++)
		{
			if (transfer_array[i] != 0)
			{
				before++;
			}
		}
		fkOptInfo->con_relid -= before;
		before = 0;
		for (int i = X + 1; i < fkOptInfo->ref_relid - 1; i++)
		{
			if (transfer_array[i] != 0)
			{
				before++;
			}
		}
		fkOptInfo->ref_relid -= before;
	}
	//×Ó²éÑ¯Éæ¼°µÄÈ«¾Örelation
	for (int i = length - 1; i > X; i--)
	{
		if (transfer_array[i] != 0)
		{
			RangeTblEntry* rte = list_nth(global_query->rtable, i);
			global_query->rtable = list_delete(global_query->rtable, list_nth(global_query->rtable, i));
			global_query->jointree->fromlist = list_delete(global_query->jointree->fromlist, list_nth(global_query->jointree->fromlist, i));
		}
	}
	RangeTblEntry* rte = (RangeTblEntry*)list_nth(global_query->rtable, X);
	dochange(rte, relname, relation, relid);
	Index index = 1;
	foreach(lc, global_query->jointree->fromlist)
	{
		RangeTblRef* rtr = (RangeTblRef*)lfirst(lc);
		rtr->rtindex = index++;
	}
	foreach(lc, global_query->targetList)
	{
		TargetEntry* tar = (TargetEntry*)lfirst(lc);
		List* target_vars = pull_var_clause(
			(Node*)tar->expr,
			PVC_RECURSE_AGGREGATES |
			PVC_RECURSE_WINDOWFUNCS |
			PVC_RECURSE_PLACEHOLDERS);
		ListCell* target_lc;
		Var* vtar = NULL;
		Index source_varno;

		foreach(target_lc, target_vars)
		{
			Var* candidate = (Var*)lfirst(target_lc);

			if (candidate->varlevelsup == 0)
			{
				vtar = candidate;
				break;
			}
		}
		if (vtar == NULL)
			continue;
		source_varno = neurqo_source_varno(vtar, length);
		if (source_varno == 0)
			ereport(ERROR,
					(errmsg("NeurQO cannot rewrite an invalid target Var"),
					 errdetail("target=%s varno=%u varnosyn=%u",
							   tar->resname != NULL ? tar->resname : "<unnamed>",
							   vtar->varno, vtar->varnosyn)));
		if (transfer_array[source_varno - 1] != 0)
		{
			tar->resorigtbl = relid;
			for (int i = 0; i < relation->rd_att->natts; i++)
			{
				if (strcmp(tar->resname, relation->rd_att->attrs[i].attname.data) == 0)
				{
					Form_pg_attribute attr =
						TupleDescAttr(relation->rd_att, i);

					tar->resorigcol = i + 1;
					vtar->varattno = i + 1;
					vtar->varno = X + 1;
					vtar->varattnosyn = vtar->varattno;
					vtar->varnosyn = vtar->varno;
					vtar->vartype = attr->atttypid;
					vtar->vartypmod = attr->atttypmod;
					vtar->varcollid = attr->attcollation;
					break;
				}
			}
		}
		else
		{
			int before = 0;
			for (int i = X + 1; i < vtar->varno; i++)
			{
				if (transfer_array[i] != 0)
				{
					before++;
				}
			}
			vtar->varno = vtar->varno - before;
			vtar->varnosyn = vtar->varno;
		}
	}
	table_close(relation, NoLock);
	return FKlist;
}

static Index
neurqo_source_varno(const Var* var, int length)
{
	if (var->varnosyn > 0 && var->varnosyn <= length)
		return var->varnosyn;
	if (var->varno > 0 && var->varno <= length)
		return var->varno;
	return 0;
}

static AttrNumber
neurqo_source_attno(const Var* var)
{
	if (var->varattnosyn > 0)
		return var->varattnosyn;
	return var->varattno;
}

/*
 * The split graph only needs simple binary join edges.  More complex
 * predicates are still preserved and pushed when all referenced relations
 * are present, but they must not be interpreted by casting arbitrary
 * expression nodes to Var.
 */
static bool
neurqo_simple_join_vars(Expr* expr, Var** left, Var** right)
{
	OpExpr* op;

	*left = NULL;
	*right = NULL;
	if (expr == NULL || !IsA(expr, OpExpr))
		return false;
	op = (OpExpr*)expr;
	if (list_length(op->args) != 2)
		return false;
	*left = neurqo_node_var((Node*)linitial(op->args));
	*right = neurqo_node_var((Node*)lsecond(op->args));
	return *left != NULL && *right != NULL &&
		(*left)->varlevelsup == 0 && (*right)->varlevelsup == 0 &&
		(*left)->varno != (*right)->varno;
}

static int
neurqo_graph_find(int* parent, int node)
{
	int root = node;

	while (parent[root] != root)
		root = parent[root];
	while (parent[node] != node)
	{
		int next = parent[node];

		parent[node] = root;
		node = next;
	}
	return root;
}

static bool
neurqo_graph_union(int* parent, unsigned char* rank, int left, int right)
{
	int left_root = neurqo_graph_find(parent, left);
	int right_root = neurqo_graph_find(parent, right);

	if (left_root == right_root)
		return false;
	if (rank[left_root] < rank[right_root])
		parent[left_root] = right_root;
	else if (rank[left_root] > rank[right_root])
		parent[right_root] = left_root;
	else
	{
		parent[right_root] = left_root;
		rank[left_root]++;
	}
	return true;
}

typedef struct NeurqoEqualityVar
{
	Index		varno;
	AttrNumber	attno;
	Oid			vartype;
	int32		vartypmod;
	Oid			varcollid;
} NeurqoEqualityVar;

static bool
neurqo_transitive_equality_vars(Expr* expr, Var** left, Var** right)
{
	OpExpr* op;
	List* interpretations;
	ListCell* lc;
	bool is_equality = false;

	*left = NULL;
	*right = NULL;
	if (expr == NULL || !IsA(expr, OpExpr))
		return false;
	op = (OpExpr*)expr;
	if (list_length(op->args) != 2 ||
		!IsA(linitial(op->args), Var) ||
		!IsA(lsecond(op->args), Var))
		return false;
	*left = (Var*)linitial(op->args);
	*right = (Var*)lsecond(op->args);
	if ((*left)->varlevelsup != 0 || (*right)->varlevelsup != 0 ||
		(*left)->varno == (*right)->varno ||
		(*left)->varattno <= 0 || (*right)->varattno <= 0 ||
		(*left)->vartype != (*right)->vartype ||
		(*left)->varcollid != (*right)->varcollid)
		return false;

	interpretations = get_op_btree_interpretation(op->opno);
	foreach(lc, interpretations)
	{
		OpBtreeInterpretation* interpretation =
			(OpBtreeInterpretation*)lfirst(lc);

		if (interpretation->strategy == BTEqualStrategyNumber &&
			interpretation->oplefttype == (*left)->vartype &&
			interpretation->oprighttype == (*right)->vartype)
		{
			is_equality = true;
			break;
		}
	}
	list_free_deep(interpretations);
	return is_equality;
}

static int
neurqo_equality_var_index(NeurqoEqualityVar* vars, int* nvars, Var* var)
{
	for (int i = 0; i < *nvars; i++)
	{
		if (vars[i].varno == var->varno &&
			vars[i].attno == var->varattno &&
			vars[i].vartype == var->vartype &&
			vars[i].vartypmod == var->vartypmod &&
			vars[i].varcollid == var->varcollid)
			return i;
	}
	vars[*nvars].varno = var->varno;
	vars[*nvars].attno = var->varattno;
	vars[*nvars].vartype = var->vartype;
	vars[*nvars].vartypmod = var->vartypmod;
	vars[*nvars].varcollid = var->varcollid;
	(*nvars)++;
	return *nvars - 1;
}

/*
 * Preserve the useful part of QuerySplit's old R-R simplification without
 * assuming that every R-R predicate is redundant.  Non-R-R equalities first
 * establish column equivalence classes.  An R-R equality is removed only when
 * its exact attributes are already connected by those equalities (or by an
 * earlier retained R-R equality), making it a true transitive cycle edge.
 *
 * This runs only after High selected split.  A stop path therefore plans the
 * untouched residual query.
 */
static int
neurqo_remove_redundant_rr_equalities(
	Query* query, bool* relationship_flags, int length)
{
	BoolExpr* and_expr;
	ListCell* lc;
	int max_vars;
	NeurqoEqualityVar* vars;
	int* parent;
	unsigned char* rank;
	int nvars = 0;
	int removed = 0;

	if (query == NULL || query->jointree == NULL ||
		query->jointree->quals == NULL ||
		!IsA(query->jointree->quals, BoolExpr))
		return 0;
	and_expr = (BoolExpr*)query->jointree->quals;
	if (and_expr->boolop != AND_EXPR || and_expr->args == NIL)
		return 0;

	max_vars = list_length(and_expr->args) * 2;
	vars = (NeurqoEqualityVar*)palloc0(
		max_vars * sizeof(NeurqoEqualityVar));
	parent = (int*)palloc(max_vars * sizeof(int));
	rank = (unsigned char*)palloc0(
		max_vars * sizeof(unsigned char));
	for (int i = 0; i < max_vars; i++)
		parent[i] = i;

	/* Build equivalence classes from predicates the old code always retained. */
	foreach(lc, and_expr->args)
	{
		Var* left;
		Var* right;
		int left_index;
		int right_index;

		if (!neurqo_transitive_equality_vars(
				(Expr*)lfirst(lc), &left, &right))
			continue;
		if (left->varno < 1 || left->varno > length ||
			right->varno < 1 || right->varno > length)
			continue;
		if (relationship_flags[left->varno - 1] &&
			relationship_flags[right->varno - 1])
			continue;
		left_index = neurqo_equality_var_index(vars, &nvars, left);
		right_index = neurqo_equality_var_index(vars, &nvars, right);
		(void)neurqo_graph_union(
			parent, rank, left_index, right_index);
	}

	/* Retain R-R bridges and remove only equality-cycle edges. */
	foreach(lc, and_expr->args)
	{
		Var* left;
		Var* right;
		int left_index;
		int right_index;

		if (!neurqo_transitive_equality_vars(
				(Expr*)lfirst(lc), &left, &right))
			continue;
		if (left->varno < 1 || left->varno > length ||
			right->varno < 1 || right->varno > length ||
			!relationship_flags[left->varno - 1] ||
			!relationship_flags[right->varno - 1])
			continue;
		left_index = neurqo_equality_var_index(vars, &nvars, left);
		right_index = neurqo_equality_var_index(vars, &nvars, right);
		if (neurqo_graph_find(parent, left_index) ==
			neurqo_graph_find(parent, right_index))
		{
			and_expr->args =
				foreach_delete_current(and_expr->args, lc);
			removed++;
		}
		else
			(void)neurqo_graph_union(
				parent, rank, left_index, right_index);
	}

	pfree(rank);
	pfree(parent);
	pfree(vars);
	return removed;
}

static void
neurqo_graph_add_oriented_edge(bool* graph, bool* relation_flags,
							   int length, int left, int right)
{
	if (query_splitting_algorithm == RelationshipCenter)
	{
		if (relation_flags[left] && !relation_flags[right])
			graph[left * length + right] = true;
		else if (!relation_flags[left] && relation_flags[right])
			graph[right * length + left] = true;
		else
		{
			graph[left * length + right] = true;
			graph[right * length + left] = true;
		}
	}
	else if (query_splitting_algorithm == EntityCenter)
	{
		if (!relation_flags[left] && relation_flags[right])
			graph[left * length + right] = true;
		else if (relation_flags[left] && !relation_flags[right])
			graph[right * length + left] = true;
		else
		{
			graph[left * length + right] = true;
			graph[right * length + left] = true;
		}
	}
}

/*
 * Build a scheduling graph without changing the SQL predicate tree.
 *
 * RelationshipCenter and EntityCenter require an acyclic decomposition graph.
 * Prefer catalog FK edges, then all other non-R-R predicates, and consider R-R
 * predicates last.  Union-find drops only cycle-closing scheduling edges; the
 * corresponding predicates remain in the Query and are evaluated either by a
 * local candidate or by a later residual query.
 */
static bool* List2Graph(bool* relation_flags, List* joinlist, List* FKlist, int length)
{
	bool* graph = (bool*)palloc0(length * length * sizeof(bool));
	ListCell* lc;

	if (query_splitting_algorithm == Minsubquery)
	{
		foreach(lc, joinlist)
		{
			Var* left;
			Var* right;

			if (!neurqo_simple_join_vars(
					(Expr*)lfirst(lc), &left, &right))
				continue;
			if (left->varno < 1 || left->varno > length ||
				right->varno < 1 || right->varno > length)
				continue;
			if (left->varno < right->varno)
				graph[(left->varno - 1) * length + right->varno - 1] = true;
			else
				graph[(right->varno - 1) * length + left->varno - 1] = true;
		}
		return graph;
	}

	if (query_splitting_algorithm == RelationshipCenter ||
		query_splitting_algorithm == EntityCenter)
	{
		int* parent = (int*)palloc(length * sizeof(int));
		unsigned char* rank =
			(unsigned char*)palloc0(length * sizeof(unsigned char));

		for (int i = 0; i < length; i++)
			parent[i] = i;

		/* Catalog FK edges have the same priority as the original QSA graph. */
		foreach(lc, FKlist)
		{
			ForeignKeyOptInfo* fk = (ForeignKeyOptInfo*)lfirst(lc);
			int con = fk->con_relid - 1;
			int ref = fk->ref_relid - 1;
			ListCell* join_lc;
			bool has_predicate = false;

			if (con < 0 || con >= length || ref < 0 || ref >= length)
				continue;
			foreach(join_lc, joinlist)
			{
				Var* left;
				Var* right;
				int left_index;
				int right_index;

				if (!neurqo_simple_join_vars(
						(Expr*)lfirst(join_lc), &left, &right))
					continue;
				left_index = left->varno - 1;
				right_index = right->varno - 1;
				if ((left_index == con && right_index == ref) ||
					(left_index == ref && right_index == con))
				{
					has_predicate = true;
					break;
				}
			}
			if (!has_predicate ||
				!neurqo_graph_union(parent, rank, con, ref))
				continue;
			if (query_splitting_algorithm == RelationshipCenter)
				graph[con * length + ref] = true;
			else
				graph[ref * length + con] = true;
		}

		/* Add non-R-R edges before optional R-R connectivity edges. */
		for (int relationship_pass = 0;
			 relationship_pass < 2;
			 relationship_pass++)
		{
			foreach(lc, joinlist)
			{
				Var* left;
				Var* right;
				int left_index;
				int right_index;
				bool is_rr;

				if (!neurqo_simple_join_vars(
						(Expr*)lfirst(lc), &left, &right))
					continue;
				if (left->varno < 1 || left->varno > length ||
					right->varno < 1 || right->varno > length)
					continue;
				left_index = left->varno - 1;
				right_index = right->varno - 1;
				is_rr = relation_flags[left_index] &&
					relation_flags[right_index];
				if (is_rr != (relationship_pass == 1))
					continue;
				if (!neurqo_graph_union(
						parent, rank, left_index, right_index))
					continue;
				neurqo_graph_add_oriented_edge(
					graph, relation_flags, length,
					left_index, right_index);
			}
		}
		pfree(rank);
		pfree(parent);
	}
	return graph;
}

//Expr is a filter clause?
static bool is_RC(Expr* expr)
{
	Var* left;
	Var* right;

	return !neurqo_simple_join_vars(expr, &left, &right);
}

//get rtable
static List* getRT_1(List* prtable, bool* graph, int length, int i, int j, Index* transfer_array)
{
	if (graph[i * length + j] == true)
	{
		List* rtable = NIL;
		RangeTblEntry* rte_i = copyObjectImpl(list_nth(prtable, i));
		RangeTblEntry* rte_j = copyObjectImpl(list_nth(prtable, j));
		rtable = lappend(rtable, rte_i);
		rtable = lappend(rtable, rte_j);
		transfer_array[i] = 1;
		transfer_array[j] = 2;
		return rtable;
	}
	return NIL;
}

static List* getRT_2(List* prtable, bool* graph, int length, int i, Index* transfer_array)
{
	Index index = 1;
	List* rtable = NIL;
	for (int j = 0; j < length; j++)
	{
		//graph[x][y]
		if (graph[i * length + j] == true)
		{
			RangeTblEntry* rte = copyObjectImpl(list_nth(prtable, j));
			rtable = lappend(rtable, rte);
			transfer_array[j] = index++;
		}
		else if (i == j)
		{
			RangeTblEntry* rte = copyObjectImpl(list_nth(prtable, i));
			rtable = lappend(rtable, rte);
			transfer_array[j] = index++;
		}
	}
	return rtable;
}

//ÕÒµ½global³ö¿Ú
static List* findvarlist(List* joinlist, Index* transfer_array, int length)
{
	ListCell* lc;
	List* reslist = NIL;

	foreach(lc, joinlist)
	{
		Expr* expr = (Expr*)lfirst(lc);
		List* vars = pull_var_clause((Node*)expr, 0);
		ListCell* var_lc;
		bool has_local = false;
		bool has_remote = false;

		foreach(var_lc, vars)
		{
			Var* var = (Var*)lfirst(var_lc);
			Index source_varno;

			if (var->varlevelsup != 0)
				continue;
			source_varno = neurqo_source_varno(var, length);
			if (source_varno == 0)
				continue;
			if (transfer_array[source_varno - 1] != 0)
				has_local = true;
			else
				has_remote = true;
		}
		if (!has_local || !has_remote)
			continue;
		foreach(var_lc, vars)
		{
			Var* var = (Var*)lfirst(var_lc);
			Index source_varno;
			AttrNumber source_attno;
			ListCell* existing_lc;
			bool append = true;

			if (var->varlevelsup != 0)
				continue;
			source_varno = neurqo_source_varno(var, length);
			source_attno = neurqo_source_attno(var);
			if (source_varno == 0 || source_attno <= 0 ||
				transfer_array[source_varno - 1] == 0)
				continue;
			foreach(existing_lc, reslist)
			{
				Var* existing = (Var*)lfirst(existing_lc);

				if (neurqo_source_varno(existing, length) == source_varno &&
					neurqo_source_attno(existing) == source_attno)
				{
					append = false;
					break;
				}
			}
			if (append)
				reslist = lappend(reslist, copyObjectImpl(var));
		}
	}
	return reslist;
}

static Query* createQuery(const Query* global_query, CommandDest dest, List* rtable, Index* transfer_array, int length)
{
	Query* query;
	query = makeNode(Query);
	query = copyObjectImpl(global_query);
	query->rtable = copyObjectImpl(rtable);
	query->jointree->fromlist = setfromlist(query->jointree->fromlist, transfer_array, length);
	List* varlist = NIL;
	if (query->jointree->quals != NULL &&
		IsA(query->jointree->quals, BoolExpr) &&
		((BoolExpr*)query->jointree->quals)->boolop == AND_EXPR)
	{
		varlist = findvarlist(
			((BoolExpr*)query->jointree->quals)->args,
			transfer_array, length);
	}
	else if (query->jointree->quals != NULL)
	{
		varlist = findvarlist(
			list_make1(query->jointree->quals),
			transfer_array, length);
	}
	query->targetList = settargetlist(global_query->rtable, rtable, dest, varlist, query->targetList, transfer_array, length);
	if (query->jointree->quals != NULL &&
		IsA(query->jointree->quals, BoolExpr) &&
		((BoolExpr*)query->jointree->quals)->boolop == AND_EXPR)
	{
		BoolExpr* quals = (BoolExpr*)query->jointree->quals;

		quals->args = setjoinlist(
			quals->args, dest, transfer_array, length);
		if (quals->args == NIL)
			query->jointree->quals = NULL;
	}
	else if (query->jointree->quals != NULL)
	{
		List* filtered = setjoinlist(
			list_make1(query->jointree->quals),
			dest, transfer_array, length);

		query->jointree->quals =
			filtered == NIL ? NULL : (Node*)linitial(filtered);
	}
	if (dest == DestRemote)
	{
		query->hasAggs = global_query->hasAggs;
	}
	else
	{
		/*
		 * The intermediate relation represents rows below the upper query
		 * operations.  settargetlist() has already replaced aggregates with
		 * the columns needed by the residual query, so retaining the parent's
		 * GROUP/ORDER/DISTINCT metadata would leave dangling sortgrouprefs and
		 * could also apply aggregation or duplicate elimination too early.
		 */
		query->returningList = NIL;
		query->groupClause = NIL;
		query->groupDistinct = false;
		query->groupingSets = NIL;
		query->havingQual = NULL;
		query->windowClause = NIL;
		query->distinctClause = NIL;
		query->sortClause = NIL;
		query->limitOffset = NULL;
		query->limitCount = NULL;
		query->rowMarks = NIL;
		query->setOperations = NULL;
		query->hasAggs = false;
		query->hasWindowFuncs = false;
		query->hasTargetSRFs = false;
		query->hasDistinctOn = false;
		query->hasForUpdate = false;
	}
	return query;
}

static List* setjoinlist(List* qualslist, CommandDest dest, Index* transfer_array, int length)
{
	ListCell* lc;

	(void)dest;
	foreach(lc, qualslist)
	{
		Expr* expr = (Expr*)lfirst(lc);
		List* vars = pull_var_clause((Node*)expr, 0);
		ListCell* var_lc;
		bool flag = true;

		foreach(var_lc, vars)
		{
			Var* var = (Var*)lfirst(var_lc);
			Index source_varno;

			if (var->varlevelsup != 0)
				continue;
			source_varno = neurqo_source_varno(var, length);
			if (source_varno == 0 ||
				transfer_array[source_varno - 1] == 0)
			{
				flag = false;
				break;
			}
		}
		if (flag)
		{
			foreach(var_lc, vars)
			{
				Var* var = (Var*)lfirst(var_lc);
				Index source_varno;

				if (var->varlevelsup != 0)
					continue;
				source_varno = neurqo_source_varno(var, length);
				var->varno = transfer_array[source_varno - 1];
				var->varnosyn = var->varno;
			}
		}
		if (!flag)
			qualslist = foreach_delete_current(qualslist, lc);
	}
	return qualslist;
}

static List* simplifyjoinlist(List* list, CommandDest dest, Index* transfer_array, bool* graph, int length)
{
	ListCell* lc;

	(void)dest;
	(void)graph;
	foreach(lc, list)
	{
		Expr* expr = (Expr*)lfirst(lc);
		List* vars = pull_var_clause((Node*)expr, 0);
		ListCell* var_lc;
		bool has_local = false;
		bool has_remote = false;

		foreach(var_lc, vars)
		{
			Var* var = (Var*)lfirst(var_lc);
			Index source_varno;

			if (var->varlevelsup != 0)
				continue;
			source_varno = neurqo_source_varno(var, length);
			if (source_varno == 0 ||
				transfer_array[source_varno - 1] == 0)
				has_remote = true;
			else
				has_local = true;
		}
		if (has_local && !has_remote)
			list = foreach_delete_current(list, lc);
	}
	return list;
}

static List* setfromlist(List* fromlist, Index* transfer_array, int length)
{
	ListCell* lc;
	foreach(lc, fromlist)
	{
		RangeTblRef* ref = (RangeTblRef*)lfirst(lc);
		if (transfer_array[ref->rtindex - 1] == 0)
		{
			fromlist = foreach_delete_current(fromlist, lc);
			continue;
		}
		ref->rtindex = transfer_array[ref->rtindex - 1];
	}
	return fromlist;
}

//varlist - global, targetlist - global
static List* settargetlist(const List* global_rtable, List* local_rtable, CommandDest dest, List* varlist, List* targetlist, Index* transfer_array, int length)
{
	ListCell* lc;
	if (dest != DestRemote)
	{
		targetlist = removeAggref(targetlist);
	}
	foreach(lc, targetlist)
	{
		TargetEntry* tar = (TargetEntry*)lfirst(lc);
		List* target_vars = pull_var_clause(
			(Node*)tar->expr,
			PVC_RECURSE_AGGREGATES |
			PVC_RECURSE_WINDOWFUNCS |
			PVC_RECURSE_PLACEHOLDERS);
		ListCell* var_lc;
		bool keep = true;

		foreach(var_lc, target_vars)
		{
			Var* var = (Var*)lfirst(var_lc);
			Index source_varno;

			if (var->varlevelsup != 0)
				continue;
			source_varno = neurqo_source_varno(var, length);
			if (source_varno == 0 ||
				transfer_array[source_varno - 1] == 0)
			{
				keep = false;
				break;
			}
		}
		if (!keep)
		{
			targetlist = foreach_delete_current(targetlist, lc);
			continue;
		}
		foreach(var_lc, target_vars)
		{
			Var* var = (Var*)lfirst(var_lc);
			Index source_varno;

			if (var->varlevelsup != 0)
				continue;
			source_varno = neurqo_source_varno(var, length);
			var->varno = transfer_array[source_varno - 1];
			var->varnosyn = var->varno;
		}
	}
	foreach(lc, varlist)
	{
		Var* var = (Var*)lfirst(lc);
		if (var != NULL)
		{
			TargetEntry* tar = makeNode(TargetEntry);
			RangeTblEntry* rte = (RangeTblEntry*)list_nth(global_rtable, var->varno - 1);
			tar->resorigtbl = rte->relid;
			int len = strlen(rte->eref->aliasname) + strlen(strVal(list_nth(rte->eref->colnames, var->varattno - 1))) + 2;
			tar->resname = (char*)palloc(len * sizeof(char));
			sprintf(tar->resname, "%s_%s", rte->eref->aliasname, strVal(list_nth(rte->eref->colnames, var->varattno - 1)));
			tar->resorigcol = var->varattno;
			if(targetlist)
				tar->resno = targetlist->length + 1;
			else
				tar->resno = 1;
			//¸Ã±äÁ¿ËùÔÚµÄ±íÖ±½Ó²ÎÓë´Ë´Îjoin
			if (transfer_array[var->varno - 1] != 0)
			{
				var->varno = transfer_array[var->varno - 1];
				var->varnosyn = var->varno;
			}
			//¸Ã±äÁ¿ËùÔÚµÄ±í¼ä½Ó²ÎÓë´Ë´Îjoin
			else
			{
				for (int i = 0; i < length; i++)
				{
					if (transfer_array[i] != 0)
					{
						var->varno = transfer_array[i];
						var->varnosyn = var->varno;
						break;
					}
				}
			}
			tar->expr = copyObjectImpl(var);
			targetlist = lappend(targetlist, tar);
		}
	}
	return targetlist;
}

//get relation foreign key
static List* grFK(List* rtable)
{
	ListCell* lc;
	List* fkey_list = NIL;
	Index relid = 0;
	foreach(lc, rtable)
	{
		relid++;
		RangeTblEntry* rte = (RangeTblEntry*)lfirst(lc);
		if (rte->relid == 0)
			continue;
		Relation relation;
		relation = table_open(rte->relid, NoLock);
		List* cachedfkeys;
		ListCell* lc1;
		cachedfkeys = RelationGetFKeyList(relation);
		foreach(lc1, cachedfkeys)
		{
			ForeignKeyCacheInfo* cachedfk = (ForeignKeyCacheInfo*)lfirst(lc1);
			Index rti;
			ListCell* lc2;
			Assert(cachedfk->conrelid == RelationGetRelid(relation));
			rti = 0;
			foreach(lc2, rtable)
			{
				RangeTblEntry* rte = (RangeTblEntry*)lfirst(lc2);
				ForeignKeyOptInfo* info;
				rti++;
				if (rte->rtekind != RTE_RELATION || rte->relid != cachedfk->confrelid)
					continue;
				if (rti == relid)
					continue;
				/* OK, let's make an entry */
				info = makeNode(ForeignKeyOptInfo);
				info->con_relid = relid;
				info->ref_relid = rti;
				info->nkeys = cachedfk->nkeys;
				memcpy(info->conkey, cachedfk->conkey, sizeof(info->conkey));
				memcpy(info->confkey, cachedfk->confkey, sizeof(info->confkey));
				memcpy(info->conpfeqop, cachedfk->conpfeqop, sizeof(info->conpfeqop));
				/* zero out fields to be filled by match_foreign_keys_to_quals */
				info->nmatched_ec = 0;
				info->nmatched_rcols = 0;
				info->nmatched_ri = 0;
				memset(info->eclass, 0, sizeof(info->eclass));
				memset(info->rinfos, 0, sizeof(info->rinfos));
				fkey_list = lappend(fkey_list, info);
			}
		}
		table_close(relation, NoLock);
	}
	return fkey_list;
}

//Is this local query the last one ?
int hasNext(bool* graph, int length)
{
	bool* temp_graph = (bool*)palloc(length * length * sizeof(bool));
	for (int i = 0; i < length * length; i++)
	{
		temp_graph[i] = graph[i];
	}
	int total_cnt = 0;
	if (query_splitting_algorithm == Minsubquery)
	{
		for (int i = 0; i < length; i++)
		{
			for (int j = i + 1; j < length; j++)
			{
				if (temp_graph[i * length + j] == true)
				{
					temp_graph[i * length + j] = false;
					temp_graph[j * length + i] = false;
					total_cnt++;
				}
			}
		}
	}
	else if (query_splitting_algorithm == RelationshipCenter || query_splitting_algorithm == EntityCenter)
	{
		for (int i = 0; i < length; i++)
		{
			int cnt = 0;
			for (int j = 0; j < length; j++)
			{
				if (i == j)
					cnt++;
				if (temp_graph[i * length + j] == true)
				{
					temp_graph[i * length + j] = false;
					temp_graph[j * length + i] = false;
					cnt++;
				}
			}
			if (cnt > 1)
				total_cnt++;
		}
	}
	pfree(temp_graph);
	temp_graph = NULL;
	return total_cnt;
}

//Change the rte's relid and name
void dochange(RangeTblEntry* rte, char* relname, Relation relation, Oid relid)
{
	rte->relid = relid;
	pfree(rte->eref->aliasname);
	rte->eref->aliasname = relname;
	list_free(rte->eref->colnames);
	rte->eref->colnames = NIL;
	for (int i = 0; i < relation->rd_att->natts; i++)
	{
		char* str = (char*)palloc((strlen(relation->rd_att->attrs[i].attname.data) + 1) * sizeof(char));
		strcpy(str, relation->rd_att->attrs[i].attname.data);
		rte->eref->colnames = lappend(rte->eref->colnames, makeString(str));
	}
	return;
}

List* makeAggref(List* targetList)
{
	List* resList = NIL;
	ListCell* lc;
	foreach(lc, targetList)
	{
		TargetEntry* old_tar = (TargetEntry*)lfirst(lc);
		Oid old_vartype = ((Var*)old_tar->expr)->vartype;
		TargetEntry* tar = makeNode(TargetEntry);
		tar->resjunk = false;
		tar->resname = old_tar->resname;
		old_tar->resname = NULL;
		tar->resno = old_tar->resno;
		tar->resorigcol = 0;
		tar->resorigtbl = 0;
		tar->ressortgroupref = 0;
		Aggref* aggref = makeNode(Aggref);
		aggref->aggargtypes = lappend_oid(NIL, old_vartype);
		aggref->aggdirectargs = NULL;
		aggref->aggdistinct = NULL;
		aggref->aggfilter = NULL;
		switch (old_vartype)
		{
			case 23:
			{
				aggref->aggfnoid = 2132;
				aggref->inputcollid = 0;
				aggref->aggcollid = 0;
				aggref->aggtype = 23;
				break;
			}
			case 25:
			{
				aggref->aggfnoid = 2145;
				aggref->inputcollid = 100;
				aggref->aggcollid = 100;
				aggref->aggtype = 25;
				break;
			}
			default:
			{
				aggref->aggfnoid = 2145;
				aggref->inputcollid = 100;
				aggref->aggcollid = 100;
				aggref->aggtype = 25;
			}
		}
		aggref->aggkind = 'n';
		aggref->agglevelsup = 0;
		aggref->aggorder = NULL;
		aggref->aggsplit = AGGSPLIT_SIMPLE;
		aggref->aggstar = false;
		aggref->aggtranstype = 0;
		aggref->aggvariadic = false;
		aggref->args = lappend(NIL, old_tar);
		aggref->location = -1;
		tar->expr = aggref;
		resList = lappend(resList, tar);
	}
	return resList;
}

List* removeAggref(List* targetList)
{
	List* resList = NIL;
	ListCell* lc;
	foreach(lc, targetList)
	{
		TargetEntry* old_tar = (TargetEntry*)lfirst(lc);
		if (old_tar->expr->type == T_Aggref)
		{
			Aggref* aggref = (Aggref*)old_tar->expr;
			TargetEntry* tar;

			/*
			 * COUNT(*) has no input column to carry through an
			 * intermediate relation.  Boundary Vars are appended below,
			 * while the original aggregate remains on the residual query.
			 */
			if (aggref->args == NIL)
				continue;
			tar = linitial(aggref->args);
			tar->resname = old_tar->resname;
			resList = lappend(resList, tar);
		}
		else
		{
			resList = lappend(resList, old_tar);
		}
	}
	return resList;
}

static Plan* find_node_with_nleaf_recursive(Plan* plan, int nleaf, int* leaf_has, int* depth)
{
	if (plan->lefttree == NULL)
	{
		*depth = *depth + 1;
		*leaf_has = 1;
		return NULL;
	}
	*depth = *depth + 1;
	int left_leaf = 0, right_leaf = 0, left_depth = *depth, right_depth = *depth;
	Plan* left_res = NULL;
	left_res = find_node_with_nleaf_recursive(plan->lefttree, nleaf, &left_leaf, &left_depth);
	Plan* right_res = NULL;
	if (plan->righttree)
		right_res = find_node_with_nleaf_recursive(plan->righttree, nleaf, &right_leaf, &right_depth);
	*leaf_has = left_leaf + right_leaf;
	if (left_res && right_res)
	{
		if (left_depth > right_depth)
		{
			*depth = left_depth;
			return left_res;
		}
		else
		{
			*depth = right_depth;
			return right_res;
		}
	}
	else if (left_res)
	{
		*depth = left_depth;;
		return left_res;
	}
	else if (right_res)
	{
		*depth = right_depth;
		return right_res;
	}
	else if (*leaf_has == nleaf)
	{
		*depth = (left_depth > right_depth) ? left_depth : right_depth;
		return plan;
	}
	else
	{
		*depth = (left_depth > right_depth) ? left_depth : right_depth;
		return NULL;
	}
}

static void walk_plantree(Plan* plan, Index* rel)
{
	Index res = 0;
	if (plan->lefttree == NULL)
	{
		res = ((Scan*)plan)->scanrelid;
		if (rel[0] == 0)
			rel[0] = res;
		else
			rel[1] = res;
	}
	if (plan->lefttree != NULL)
		walk_plantree(plan->lefttree, rel);
	if (plan->righttree != NULL)
		walk_plantree(plan->righttree, rel);
	return;
}

int tarfunc(Index* rels, PlannedStmt* new, PlannedStmt* old)
{
	if(old == NULL)
		return NEWBETTER;
	if (new->planTree->plan_rows > 10000000)
	{
		return OLDBETTER;
	}
	if (order_decision == only_cost)
	{
		if (old->planTree->total_cost < new->planTree->total_cost)
		{
			return OLDBETTER;
		}
		else
		{
			return NEWBETTER;
		}
	}
	if (order_decision == only_row)
	{
		if (old->planTree->plan_rows < new->planTree->plan_rows)
		{
			return OLDBETTER;
		}
		else
		{
			return NEWBETTER;
		}
	}
	double fac_old, fac_new;
	if (order_decision == hybrid_row)
	{
		if (new->planTree->plan_rows > 1)
			fac_new = new->planTree->plan_rows;
		else
			fac_new = 1;
		if (old->planTree->plan_rows > 1)
			fac_old = old->planTree->plan_rows;
		else
			fac_old = 1;
		if (fac_new / fac_old > old->planTree->total_cost / new->planTree->total_cost)
		{
			return OLDBETTER;
		}
		else
		{
			return NEWBETTER;
		}
	}
	else if (order_decision == hybrid_sqrt)
	{
		double fac_old, fac_new;
		if (new->planTree->plan_rows > 1)
			fac_new = sqrt(new->planTree->plan_rows);
		else
			fac_new = 1;
		if (old->planTree->plan_rows > 1)
			fac_old = sqrt(old->planTree->plan_rows);
		else
			fac_old = 1;
		if (fac_new / fac_old > old->planTree->total_cost / new->planTree->total_cost)
		{
			return OLDBETTER;
		}
		else
		{
			return NEWBETTER;
		}
	}
	else if (order_decision == hybrid_log)
	{
		if (new->planTree->plan_rows > 1)
			fac_new = log(new->planTree->plan_rows) / log(2);
		else
			fac_new = 1;
		if (old->planTree->plan_rows > 1)
			fac_old = log(old->planTree->plan_rows) / log(2);
		else
			fac_old = 1;
		if (fac_new / fac_old > old->planTree->total_cost / new->planTree->total_cost)
		{
			return OLDBETTER;
		}
		else
		{
			return NEWBETTER;
		}
	}
	else if (order_decision == global_view)
	{
		ListCell* lc;
		bool flag = false;
		foreach(lc, new->rtable)
		{
			RangeTblEntry* rte = (RangeTblEntry*)lfirst(lc);
			if (rte->relid == rels[0])
			{
				flag = true;
				break;
			}
		}
		if (!flag)
			return OLDBETTER;
		flag = false;
		foreach(lc, new->rtable)
		{
			RangeTblEntry* rte = (RangeTblEntry*)lfirst(lc);
			if (rte->relid == rels[1])
			{
				flag = true;
				break;
			}
		}
		if (!flag)
			return OLDBETTER;
		return NEWBETTER;
	}
}
