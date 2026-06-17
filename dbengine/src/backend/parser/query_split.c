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

#include "fe_utils/simple_list.h"
#include "nodes/nodeFuncs.h"
#include "commands/event_trigger.h"
#include "commands/portalcmds.h"
#include "utils/relmapper.h"
#include "commands/vacuum.h"
#include "lib/stringinfo.h"
#include "parser/parse_func.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/ruleutils.h"
#include "storage/fd.h"
#include "storage/buf_internals.h"
#include "parser/parse_relation.h"		/* addRTEPermissionInfo (PG16) */

#define NEWBETTER 1
#define OLDBETTER 2
#define NEURQO_MAX_LIP_FILTERS 10
#define NEURQO_SEARCH_ABS_MAX_RELS 16
#define NEURQO_SEARCH_ABS_MAX_K 16
#define NEURQO_AJA_PLAN_MAX_NODES 96

//Create a local query
static Query* createQuery(const Query* querytree, CommandDest dest, List* rtable, Index* transfer_array, int length);
//change the RangeTblEntry relid to the new one
static void dochange(RangeTblEntry* rte, char* relname, Relation relation, Oid relid);
//change the NullTest clause args to the new one
static bool doNullTestTransfor(NullTest* expr, Index* transfer_array);
//change the OpExpr clause args to the new one
static bool doOpExprTransfor(OpExpr* expr, Index* transfer_array);
//change the ScalarArrayOpExpr clause args to the new one
static bool doScalarArrayOpExprTransfor(ScalarArrayOpExpr* expr, Index* transfer_array);
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
// Is this expr refer to two relationship table ?
static bool is_2relationship(OpExpr * opexpr, bool* is_relationship, int length);
//Is this a Entity-to-Relationship Join ?
static bool is_ER(OpExpr* opexpr, bool* is_relationship, int length);
//Is a foreign key join ?
static bool is_FK(OpExpr* opexpr, List* fklist);
//Is a restrict clause ?
static bool is_RC(Expr* expr);
//Transefer jointree to graph
static bool* List2Graph(bool* is_relationship, List* joinlist, List* FKlist, int length);
//Make a aggregation function as result
static List* removeAggref(List* targetList);
//give the new value to some var, prepare for the next subquery
static List* Prepare4Next(Query* global_query, Index* transfer_array, DR_intorel* receiver, PlannedStmt* plannedstmt,char* relname, List* FKlist);
static void Recon(char* query_string, CommandTag commandTag, Node* pstmt, Query* ori_query, QueryCompletion* completionTag);
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
static List* QSExecutor(char* query_string, CommandTag commandTag, Node* pstmt, PlannedStmt* plannedstmt, CommandDest dest, char* relname, QueryCompletion* completionTag, Query* querytree, Index* transfer_array, List* FKlist, MemoryContext oldcontext);
//find the subquery with lowest cost to be executed
static PlannedStmt* QSOptimizer(Query* global_query, bool* graph, Index* transfer_array, int length);
static Plan* find_node_with_nleaf_recursive(Plan* plan, int nleaf, int* leaf_has, int* depth);
static void walk_plantree(Plan* plan, Index* rel);
static PlannedStmt* neurqo_plan(Query* q, int cursorOptions, bool apply_lip);
static PlannedStmt* neurqo_plan_direct(Query* q, int cursorOptions,
									   bool apply_lip,
									   const char* hint_query_string,
									   bool log_hint);
static char* neurqo_build_planner_hint(Query* q);
static char* neurqo_build_search_hint(Query* q);
static char* neurqo_build_aja_hint(Query* q, const char* search_hint_body);
static const char* neurqo_order_decision_name(int mode);
static bool neurqo_search_enabled(void);
static bool neurqo_aja_enabled(void);
static bool neurqo_lip_enabled(void);
static bool neurqo_policy_round(Query* q, const char* query_string,
								int round, int length, int remaining,
								bool* stop_now, double* policy_ms,
								char** state_json_out);
static double neurqo_now_ms(void);
static bool neurqo_apply_lip(Query* q, double* lip_ms, int* lip_filters);
static char* neurqo_make_hint_query(const char* first_hint,
									const char* second_hint);
static char* neurqo_build_round_state(Query* q, const char* query_string,
									  int round, int length, int remaining);
static void neurqo_append_plan_state_fields(Query* q, StringInfo out,
											const char* search_hint_body);
static void neurqo_plan_summary(Plan* plan, int depth, int* nnodes,
								int* njoins, int* nscans, int* max_depth);
static void neurqo_append_aliases_json(Query* q, StringInfo out);
static void neurqo_append_plan_json(Plan* plan, Query* q, StringInfo out,
									int* nnodes);
static void neurqo_log_trajectory_event(const char* phase, int round,
										const char* state_json, bool stop_now,
										double policy_ms, double planning_ms,
										double execution_ms, double total_ms,
										const char* result);

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
static char neurqo_current_search_strategy[64] = "";
static char neurqo_current_execution_action[64] = "";
static char neurqo_current_lip_action[64] = "";
static int neurqo_current_search_k = 0;
//the number of subquery
static int queryId = 0;
static uint64 neurqo_run_seq = 0;
static uint64 neurqo_current_run_id = 0;
//where to send the result, to the client end or temporary table
CommandDest mydest;
Index* transfer_array = NULL;

typedef struct NeurqoPolicyAction
{
	char action[64];
	bool stop;
	bool has_order_decision;
	int order_decision;
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

typedef struct NeurqoLipFilter
{
	int			filter_id;
	Var		   *build_var;
	Var		   *probe_var;
} NeurqoLipFilter;

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
						 int round, int length, int remaining)
{
	StringInfoData state;
	ListCell* lc;
	bool first = true;

	initStringInfo(&state);
	appendStringInfo(&state,
					 "{\"pid\":%d,\"run_id\":" UINT64_FORMAT
					 ",\"request_type\":\"round\",\"round\":%d,"
					 "\"base_rels\":%d,\"remaining_splits\":%d,"
					 "\"algorithm\":%d,\"order_decision\":\"%s\","
					 "\"lip_action\":\"%s\",\"sql\":",
					 MyProcPid, neurqo_current_run_id, round, length, remaining,
					 query_splitting_algorithm,
					 neurqo_order_decision_name(order_decision),
					 neurqo_lip_enabled() ? neurqo_current_lip_action : "none");
	neurqo_append_json_string(&state, query_string);
	appendStringInfoString(&state, ",\"relations\":[");
	foreach(lc, q->rtable)
	{
		RangeTblEntry* rte = (RangeTblEntry*)lfirst(lc);
		const char* alias;
		char* relname = NULL;

		if (rte->rtekind != RTE_RELATION)
			continue;
		alias = rte->eref ? rte->eref->aliasname : NULL;
		relname = get_rel_name(rte->relid);
		if (!first)
			appendStringInfoChar(&state, ',');
		first = false;
		appendStringInfoString(&state, "{\"alias\":");
		neurqo_append_json_string(&state, alias);
		appendStringInfoString(&state, ",\"relname\":");
		neurqo_append_json_string(&state, relname ? relname : alias);
		appendStringInfo(&state, ",\"relid\":%u}", rte->relid);
		if (relname)
			pfree(relname);
	}
	appendStringInfoString(&state, "],");
	neurqo_append_plan_state_fields(q, &state, NULL);
	appendStringInfoString(&state, "}");
	return state.data;
}

static void
neurqo_log_trajectory_event(const char* phase, int round,
							const char* state_json, bool stop_now,
							double policy_ms, double planning_ms,
							double execution_ms, double total_ms,
							const char* result)
{
	FILE* fp;
	StringInfoData line;

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
	appendStringInfoString(&line, ",\"action\":{");
	appendStringInfoString(&line, "\"order_decision\":");
	neurqo_append_json_string(&line, neurqo_order_decision_name(order_decision));
	appendStringInfoString(&line, ",\"search_strategy\":");
	neurqo_append_json_string(&line,
							  neurqo_search_enabled() ?
							  neurqo_current_search_strategy : "default");
	appendStringInfo(&line, ",\"search_k\":%d",
					 neurqo_current_search_k > 0 ?
					 neurqo_current_search_k : neurqo_search_topk);
	appendStringInfoString(&line, ",\"execution_action\":");
	neurqo_append_json_string(&line,
							  neurqo_aja_enabled() ?
							  neurqo_current_execution_action : "none");
	appendStringInfoString(&line, ",\"lip_action\":");
	neurqo_append_json_string(&line,
							  neurqo_lip_enabled() ?
							  neurqo_current_lip_action : "none");
	appendStringInfoString(&line, "},\"timing_ms\":{");
	appendStringInfo(&line,
					 "\"policy\":%.3f,\"planning\":%.3f,"
					 "\"execution\":%.3f,\"total\":%.3f}",
					 policy_ms, planning_ms, execution_ms, total_ms);
	appendStringInfoString(&line, ",\"result\":");
	neurqo_append_json_string(&line, result);
	appendStringInfoChar(&line, '}');

	fputs(line.data, fp);
	fputc('\n', fp);
	FreeFile(fp);
	pfree(line.data);
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
		(strcmp(neurqo_current_execution_action, "aja") == 0 ||
		 strcmp(neurqo_current_execution_action, "hashjoin") == 0 ||
		 strcmp(neurqo_current_execution_action, "nestloop") == 0);
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

static bool
neurqo_policy_round(Query* q, const char* query_string,
					int round, int length, int remaining,
					bool* stop_now, double* policy_ms,
					char** state_json_out)
{
	StringInfoData resp;
	NeurqoPolicyAction act;
	char errbuf[256];
	double t0;
	double t1;
	bool ok;
	char* state_json;

	*stop_now = false;
	*policy_ms = 0.0;
	if (state_json_out != NULL)
		*state_json_out = NULL;
	state_json = neurqo_build_round_state(q, query_string, round, length,
										  remaining);

	initStringInfo(&resp);
	t0 = neurqo_now_ms();
	ok = neurqo_http_post(neurqo_server_url, state_json, &resp,
						  errbuf, sizeof(errbuf));
	t1 = neurqo_now_ms();
	*policy_ms = t1 - t0;

	if (!ok)
	{
		elog(WARNING, "[neurqo] run=" UINT64_FORMAT " round %d: AI server call failed (%s); fallback to local RCenter policy",
			 neurqo_current_run_id, round, errbuf);
		pfree(resp.data);
		if (state_json_out != NULL)
			*state_json_out = state_json;
		else
			pfree(state_json);
		return false;
	}

	neurqo_parse_policy_action(resp.data, &act);
	if (act.has_order_decision)
		order_decision = act.order_decision;
	if (act.has_search_strategy)
		snprintf(neurqo_current_search_strategy,
				 sizeof(neurqo_current_search_strategy),
				 "%s", act.search_strategy);
	if (act.has_search_k && act.search_k > 0)
		neurqo_current_search_k = act.search_k;
	if (act.has_execution_action)
		snprintf(neurqo_current_execution_action,
				 sizeof(neurqo_current_execution_action),
				 "%s", act.execution_action);
	if (act.has_lip_action)
		snprintf(neurqo_current_lip_action,
				 sizeof(neurqo_current_lip_action),
				 "%s", act.lip_action);
	else if (strcmp(act.action, "lip") == 0 ||
			 strcmp(act.action, "lip_aja") == 0 ||
			 strcmp(act.action, "lip_search") == 0)
		snprintf(neurqo_current_lip_action,
				 sizeof(neurqo_current_lip_action),
				 "full");
	*stop_now = act.stop;

	elog(LOG, "[neurqo] run=" UINT64_FORMAT " round %d: state=%s action=%s stop=%d order_decision=%s search_strategy=%s search_k=%d execution_action=%s lip_action=%s note=\"%s\" policy_ms=%.2f",
		 neurqo_current_run_id, round, state_json, act.action, act.stop ? 1 : 0,
		 neurqo_order_decision_name(order_decision),
		 neurqo_search_enabled() ? neurqo_current_search_strategy : "default",
		 neurqo_current_search_k > 0 ? neurqo_current_search_k : neurqo_search_topk,
		 neurqo_aja_enabled() ? neurqo_current_execution_action : "none",
		 neurqo_lip_enabled() ? neurqo_current_lip_action : "none",
		 act.note, *policy_ms);

	pfree(resp.data);
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
	if (var == NULL || var->varlevelsup != 0 ||
		var->varno == 0 || var->varno > list_length(q->rtable))
		return NULL;

	RangeTblEntry* rte = (RangeTblEntry*)list_nth(q->rtable, var->varno - 1);

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

static int
neurqo_collect_lip_filters(Query* q, List* clauses,
						   NeurqoLipFilter* filters)
{
	bool* has_local_restrict;
	ListCell* lc;
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

		if (!neurqo_is_int4_equi_join((Expr*)lfirst(lc), &left, &right))
			continue;
		if (neurqo_rte_for_var(q, left) == NULL ||
			neurqo_rte_for_var(q, right) == NULL)
			continue;

		left_local = left->varno <= nrtables && has_local_restrict[left->varno];
		right_local = right->varno <= nrtables && has_local_restrict[right->varno];
		if (selective && !left_local && !right_local)
			continue;

		build_var = left;
		probe_var = right;
		if (right_local && !left_local)
		{
			build_var = right;
			probe_var = left;
		}
		else if (left_local == right_local &&
				 !neurqo_var_att_is_id(q, left) &&
				 neurqo_var_att_is_id(q, right))
		{
			build_var = right;
			probe_var = left;
		}

		if (neurqo_lip_filter_exists(filters, nfilters, build_var, probe_var))
			continue;
		filters[nfilters].filter_id = nfilters;
		filters[nfilters].build_var = (Var*)copyObjectImpl(build_var);
		filters[nfilters].probe_var = (Var*)copyObjectImpl(probe_var);
		nfilters++;
		if (nfilters >= NEURQO_MAX_LIP_FILTERS)
			break;
	}

	pfree(has_local_restrict);
	return nfilters;
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
					 int nfilters)
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

	for (i = 0; i < nfilters; i++)
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
neurqo_apply_lip(Query* q, double* lip_ms, int* lip_filters)
{
	List* clauses = NIL;
	NeurqoLipFilter filters[NEURQO_MAX_LIP_FILTERS];
	Oid probe_funcid;
	int nfilters;
	int i;
	double t0 = neurqo_now_ms();

	*lip_ms = 0.0;
	*lip_filters = 0;
	if (!neurqo_lip_enabled() || q == NULL || q->jointree == NULL)
		return true;

	neurqo_flatten_and_clauses(q->jointree->quals, &clauses);
	nfilters = neurqo_collect_lip_filters(q, clauses, filters);
	if (nfilters <= 0)
	{
		elog(LOG, "[neurqo] run=" UINT64_FORMAT " LIP skipped: no eligible int4 equi-join filters mode=%s",
			 neurqo_current_run_id, neurqo_current_lip_action);
		return true;
	}

	if (!neurqo_lip_run_setup(q, clauses, filters, nfilters))
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

	for (i = 0; i < nfilters; i++)
		neurqo_lip_add_probe_qual(q, probe_funcid, &filters[i]);

	*lip_ms = neurqo_now_ms() - t0;
	*lip_filters = nfilters;
	elog(LOG, "[neurqo] run=" UINT64_FORMAT " apply LIP: mode=%s filters=%d lip_ms=%.2f",
		 neurqo_current_run_id, neurqo_current_lip_action, nfilters, *lip_ms);
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
		neurqo_apply_lip(q, &lip_ms, &lip_filters);
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

		neurqo_apply_lip(q, &lip_ms, &lip_filters);
	}
	r = neurqo_plan_direct(q, cursorOptions, false, hint_query_string, true);
	if (hint_query_string != NULL)
		pfree(hint_query_string);
	return r;
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

		pfree(map);
		return local_query;
	}
}

static double
neurqo_subset_cardinality(Query* q, NeurqoSearchRel* rels, int nrels,
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
neurqo_build_topk_leading_hint(Query* q)
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

	if (max_rels <= 0 || max_rels > NEURQO_SEARCH_ABS_MAX_RELS)
		max_rels = NEURQO_SEARCH_ABS_MAX_RELS;
	nrels = neurqo_collect_search_rels(q, rels, NEURQO_SEARCH_ABS_MAX_RELS,
									   &all_rte_relation);
	if (nrels < 2)
		return NULL;
	if (!all_rte_relation || nrels > max_rels)
	{
		elog(LOG, "[neurqo] run=" UINT64_FORMAT " Search top-k skipped: nrels=%d all_relation=%d max_rels=%d; fallback left_deep",
			 neurqo_current_run_id, nrels, all_rte_relation ? 1 : 0, max_rels);
		return neurqo_build_left_deep_leading_hint(q);
	}

	neurqo_collect_join_edges(q, rels, nrels, edges);
	full_mask = (((uint64)1) << nrels) - 1;
	nmasks = full_mask + 1;
	cells = (NeurqoSearchCell*)palloc0(sizeof(NeurqoSearchCell) * nmasks);

	for (i = 0; i < nrels; i++)
	{
		uint64 mask = ((uint64)1) << i;

		neurqo_search_cell_add(&cells[mask], 0.0, rels[i].alias, k);
		cells[mask].card_valid = true;
		cells[mask].card_rows = 1.0;
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
				join_rows = neurqo_subset_cardinality(q, rels, nrels, cells,
													  mask);
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
		elog(LOG, "[neurqo] run=" UINT64_FORMAT " Search top-k found no connected DP order; fallback left_deep",
			 neurqo_current_run_id);
		return neurqo_build_left_deep_leading_hint(q);
	}

	for (i = 0; i < cells[full_mask].nentries; i++)
	{
		char* search_hint = psprintf("Leading(%s)", cells[full_mask].entries[i].leading);
		char* hint_query = neurqo_make_hint_query(search_hint, NULL);
		PlannedStmt* planned = neurqo_plan_direct(copyObjectImpl(q),
												  CURSOR_OPT_PARALLEL_OK,
												  false, hint_query, false);
		double cost = planned && planned->planTree ?
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
		}
		pfree(search_hint);
		if (hint_query != NULL)
			pfree(hint_query);
	}

	if (best_leading == NULL)
		return neurqo_build_left_deep_leading_hint(q);

	elog(LOG, "[neurqo] run=" UINT64_FORMAT " Search top-k applied: strategy=%s k=%d nrels=%d candidates=%d best_physical_cost=%.2f search_ms=%.2f leading=%s",
		 neurqo_current_run_id, neurqo_current_search_strategy, k, nrels,
		 cells[full_mask].nentries, best_cost, neurqo_now_ms() - t0,
		 best_leading);
	return psprintf("Leading(%s)", best_leading);
}

static char*
neurqo_build_search_hint(Query* q)
{
	if (!neurqo_search_enabled())
		return NULL;
	if (strcmp(neurqo_current_search_strategy, "left_deep") == 0)
		return neurqo_build_left_deep_leading_hint(q);
	return neurqo_build_topk_leading_hint(q);
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

static void
neurqo_append_plan_state_fields(Query* q, StringInfo out,
								const char* search_hint_body)
{
	PlannedStmt* baseline;
	char* search_hint_query = NULL;
	int nnodes = 0;
	int njoins = 0;
	int nscans = 0;
	int max_depth = 0;
	int plan_json_nodes = 0;
	double t0 = neurqo_now_ms();

	if (search_hint_body != NULL && search_hint_body[0] != '\0')
		search_hint_query = neurqo_make_hint_query(search_hint_body, NULL);

	baseline = neurqo_plan_direct(copyObjectImpl(q), CURSOR_OPT_PARALLEL_OK,
								  false, search_hint_query, false);
	if (search_hint_query != NULL)
		pfree(search_hint_query);

	if (baseline == NULL || baseline->planTree == NULL)
	{
		appendStringInfoString(out, "\"plan_available\":false");
		return;
	}

	neurqo_plan_summary(baseline->planTree, 1, &nnodes, &njoins, &nscans,
						&max_depth);
	appendStringInfo(out,
					 "\"plan_available\":true,"
					 "\"plan_total_cost\":%.2f,\"plan_rows\":%.0f,"
					 "\"plan_width\":%d,\"plan_state_ms\":%.3f,"
					 "\"plan_summary\":{\"nodes\":%d,\"joins\":%d,"
					 "\"scans\":%d,\"max_depth\":%d},\"aliases\":",
					 baseline->planTree->total_cost,
					 baseline->planTree->plan_rows,
					 baseline->planTree->plan_width,
					 neurqo_now_ms() - t0,
					 nnodes, njoins, nscans, max_depth);
	neurqo_append_aliases_json(q, out);
	appendStringInfoString(out, ",\"plan_json\":");
	neurqo_append_plan_json(baseline->planTree, q, out, &plan_json_nodes);
}

static char*
neurqo_build_aja_hint(Query* q, const char* search_hint_body)
{
	PlannedStmt* baseline;
	char* search_hint_query = NULL;
	StringInfoData state;
	StringInfoData resp;
	NeurqoPolicyAction act;
	char errbuf[256];
	int nnodes = 0;
	int njoins = 0;
	int nscans = 0;
	int max_depth = 0;
	int plan_json_nodes = 0;
	double t0 = neurqo_now_ms();
	bool ok;

	if (strcmp(neurqo_current_execution_action, "hashjoin") == 0 ||
		strcmp(neurqo_current_execution_action, "nestloop") == 0 ||
		strcmp(neurqo_current_execution_action, "mergejoin") == 0)
		return neurqo_build_join_method_hint(q, neurqo_current_execution_action);
	if (strcmp(neurqo_current_execution_action, "aja") != 0)
		return NULL;

	search_hint_query = neurqo_make_hint_query(search_hint_body, NULL);
	baseline = neurqo_plan_direct(copyObjectImpl(q), CURSOR_OPT_PARALLEL_OK,
								  false, search_hint_query, false);
	if (search_hint_query != NULL)
		pfree(search_hint_query);
	if (baseline == NULL || baseline->planTree == NULL)
		return neurqo_build_join_method_hint(q, "hashjoin");

	neurqo_plan_summary(baseline->planTree, 1, &nnodes, &njoins, &nscans,
						&max_depth);
	initStringInfo(&state);
	appendStringInfo(&state,
					 "{\"request_type\":\"aja\",\"pid\":%d,"
					 "\"run_id\":" UINT64_FORMAT ",\"base_rels\":%d,"
					 "\"plan_total_cost\":%.2f,\"plan_rows\":%.0f,"
					 "\"plan_summary\":{\"nodes\":%d,\"joins\":%d,"
					 "\"scans\":%d,\"max_depth\":%d},\"search_hint\":",
					 MyProcPid, neurqo_current_run_id,
					 list_length(q->rtable),
					 baseline->planTree->total_cost,
					 baseline->planTree->plan_rows,
					 nnodes, njoins, nscans, max_depth);
	neurqo_append_json_string(&state, search_hint_body);
	appendStringInfoString(&state, ",\"aliases\":");
	neurqo_append_aliases_json(q, &state);
	appendStringInfoString(&state, ",\"plan\":");
	neurqo_append_plan_json(baseline->planTree, q, &state, &plan_json_nodes);
	appendStringInfoChar(&state, '}');

	initStringInfo(&resp);
	ok = neurqo_http_post(neurqo_server_url, state.data, &resp,
						  errbuf, sizeof(errbuf));
	if (!ok)
	{
		elog(WARNING, "[neurqo] run=" UINT64_FORMAT " AJA server call failed (%s); fallback HashJoin hint",
			 neurqo_current_run_id, errbuf);
		pfree(state.data);
		pfree(resp.data);
		return neurqo_build_join_method_hint(q, "hashjoin");
	}

	neurqo_parse_policy_action(resp.data, &act);
	elog(LOG, "[neurqo] run=" UINT64_FORMAT " AJA baseline_plan cost=%.2f rows=%.0f joins=%d scans=%d depth=%d server_resp=%s aja_ms=%.2f",
		 neurqo_current_run_id, baseline->planTree->total_cost,
		 baseline->planTree->plan_rows, njoins, nscans, max_depth,
		 resp.data, neurqo_now_ms() - t0);
	pfree(state.data);
	pfree(resp.data);

	if (act.has_aja_hint &&
		pg_strcasecmp(act.aja_hint, "none") != 0 &&
		pg_strcasecmp(act.aja_hint, "default") != 0)
	{
		if (strchr(act.aja_hint, '(') != NULL)
			return pstrdup(act.aja_hint);
		return neurqo_build_join_method_hint(q, act.aja_hint);
	}
	if (act.has_join_method &&
		pg_strcasecmp(act.join_method, "none") != 0 &&
		pg_strcasecmp(act.join_method, "default") != 0)
		return neurqo_build_join_method_hint(q, act.join_method);

	return neurqo_build_join_method_hint(q, "hashjoin");
}

static char*
neurqo_build_planner_hint(Query* q)
{
	char* search_hint = NULL;
	char* aja_hint = NULL;
	char* hint_query = NULL;

	if (neurqo_search_enabled())
		search_hint = neurqo_build_search_hint(q);
	if (neurqo_aja_enabled())
		aja_hint = neurqo_build_aja_hint(q, search_hint);

	hint_query = neurqo_make_hint_query(aja_hint, search_hint);
	if (search_hint != NULL)
		pfree(search_hint);
	if (aja_hint != NULL)
		pfree(aja_hint);
	return hint_query;
}

//The interface
void doQSparse(const char* query_string, CommandTag commandTag, Node* pstmt, Query* querytree, QueryCompletion* completionTag)
{
	List* FKlist = NIL;
	neurqo_current_run_id = ++neurqo_run_seq;
	neurqo_current_search_strategy[0] = '\0';
	neurqo_current_execution_action[0] = '\0';
	neurqo_current_lip_action[0] = '\0';
	neurqo_current_search_k = 0;
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
		if (rte->relkind != RELKIND_RELATION)
		{
			MemoryContext oldcontext = MemoryContextSwitchTo(MessageContext);
			plannedstmt = neurqo_plan(querytree, CURSOR_OPT_PARALLEL_OK, true);
			QSExecutor(query_string, commandTag, pstmt, plannedstmt, DestRemote, NULL, completionTag, querytree, NULL, NIL, oldcontext);
			return;
		}
		length++;
	}
	if (length <= 2)
	{
		MemoryContext oldcontext = MemoryContextSwitchTo(MessageContext);
		plannedstmt = neurqo_plan(querytree, CURSOR_OPT_PARALLEL_OK, true);
		QSExecutor(query_string, commandTag, pstmt, plannedstmt, DestRemote, NULL, completionTag, querytree, NULL, NIL, oldcontext);
		return;
	}
	//split parent query by foreign key
	Recon(query_string, commandTag, pstmt, querytree, completionTag);

	return;
}

//remove Redundant Join
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
	if (querytree->jointree->quals == NULL)
		return;
	//SQL where clause
	switch (querytree->jointree->quals->type)
	{
		case T_BoolExpr:
		{
			BoolExpr* expr = (BoolExpr*)querytree->jointree->quals;
			if (expr == NULL)
				return;
			//expression list in the SQL where clause
			List* where = expr->args;
			//remove the redundant expression in expression list
			foreach(lc, where)
			{
				//is this expression a filter clause ?
				if (is_RC(lfirst(lc)))
				{
					continue;
				}
				//is this expression contian two relationship table ?
				if (is_2relationship(lfirst(lc), is_relationship, length))
				{
					//if yes, remove it from expression list
					where = foreach_delete_current(where, lc);
					continue;
				}
			}
			break;
		}
		case T_OpExpr:
			break;
	}
	return;
}

static void Recon(char* query_string, CommandTag commandTag, Node* pstmt, Query* ori_query, QueryCompletion* completionTag)
{
	MemoryContext oldcontext = MemoryContextSwitchTo(MessageContext);
	Query* global_query = copyObjectImpl(ori_query);
	PlannedStmt* plannedstmt = NULL;
	if (global_query->commandType == CMD_UTILITY)
	{
		plannedstmt = QSOptimizer(global_query, NULL, NULL, 0);
		QSExecutor(query_string, commandTag, pstmt, plannedstmt, DestRemote, NULL, completionTag, NULL, NULL, NIL, oldcontext);
		return;
	}
	int length = global_query->rtable->length;
	if (length == 1)
	{
		plannedstmt = QSOptimizer(global_query, NULL, NULL, length);
		QSExecutor(query_string, commandTag, pstmt, plannedstmt, DestRemote, NULL, completionTag, NULL, NULL, NIL, oldcontext);
		return;
	}
	List* RClist = NIL;
	List* Joinlist = NIL;
	List* WhereClause = NIL;
	switch (global_query->jointree->quals->type)
	{
		case T_BoolExpr:
		{
			BoolExpr* expr = (BoolExpr*)global_query->jointree->quals;
			WhereClause = expr->args;
			break;
		}
		case T_OpExpr:
		{
			WhereClause = lappend(WhereClause, global_query->jointree->quals);
			break;
		}
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
	while (true)
	{
		bool stop_now = false;
		double policy_ms = 0.0;
		double optimize_ms = 0.0;
		double exec_ms = 0.0;
		double round_start = neurqo_now_ms();
		double t0;
		int remaining = hasNext(graph, length);
		char* state_json = NULL;

		if (remaining <= 0)
			break;
		if (policy_available)
			policy_available = neurqo_policy_round(global_query, query_string,
												   round, length, remaining,
												   &stop_now, &policy_ms,
												   &state_json);
		if (round >= neurqo_max_rounds)
		{
			stop_now = true;
			elog(LOG, "[neurqo] run=" UINT64_FORMAT " round %d: reached neurqo.max_rounds=%d; finishing residual query",
				 neurqo_current_run_id, round, neurqo_max_rounds);
		}
		if (stop_now)
		{
			t0 = neurqo_now_ms();
			plannedstmt = neurqo_plan(global_query, CURSOR_OPT_PARALLEL_OK, true);
			optimize_ms = neurqo_now_ms() - t0;
			t0 = neurqo_now_ms();
			QSExecutor(query_string, commandTag, pstmt, plannedstmt, DestRemote,
					   NULL, completionTag, global_query, transfer_array, FKlist,
					   oldcontext);
			exec_ms = neurqo_now_ms() - t0;
			neurqo_log_trajectory_event("final", round, state_json, stop_now,
										policy_ms, optimize_ms, exec_ms,
										neurqo_now_ms() - round_start,
										"remote");
			elog(LOG, "[neurqo] run=" UINT64_FORMAT " round %d: final residual executed policy_ms=%.2f planning_ms=%.2f execution_ms=%.2f total_ms=%.2f",
				 neurqo_current_run_id, round, policy_ms, optimize_ms, exec_ms,
				 neurqo_now_ms() - round_start);
			if (state_json != NULL)
				pfree(state_json);
			break;
		}

		t0 = neurqo_now_ms();
		plannedstmt = QSOptimizer(global_query, graph, transfer_array, length);
		optimize_ms = neurqo_now_ms() - t0;
		if (plannedstmt == NULL)
		{
			if (state_json != NULL)
				pfree(state_json);
			break;
		}
		queryId++;
		char* relname = NULL;
		//Should we output the result or save it as a temporary table
		if (mydest == DestIntoRel)
		{
			relname = palloc(7 * sizeof(char));
			sprintf(relname, "temp%d", queryId);
		}
		//Execute the subquery and do some change for next subquery creation
		t0 = neurqo_now_ms();
		FKlist = QSExecutor(query_string, commandTag, pstmt, plannedstmt, mydest, relname, completionTag, global_query, transfer_array, FKlist, oldcontext);
		exec_ms = neurqo_now_ms() - t0;
		neurqo_log_trajectory_event(mydest == DestIntoRel ? "split" : "final",
									round, state_json,
									mydest == DestRemote, policy_ms,
									optimize_ms, exec_ms,
									neurqo_now_ms() - round_start,
									mydest == DestIntoRel ? relname : "remote");
		elog(LOG, "[neurqo] run=" UINT64_FORMAT " round %d: apply split result=%s policy_ms=%.2f split_planning_ms=%.2f execution_rewrite_ms=%.2f total_ms=%.2f",
			 neurqo_current_run_id, round,
			 mydest == DestIntoRel ? relname : "remote",
			 policy_ms, optimize_ms, exec_ms, neurqo_now_ms() - round_start);
		if (state_json != NULL)
			pfree(state_json);
		//finish_xact_command();
		if (mydest == DestRemote)
		{
			break;
		}
		switch (global_query->jointree->quals->type)
		{
			case T_BoolExpr:
			{
				BoolExpr* expr = (BoolExpr*)global_query->jointree->quals;
				WhereClause = expr->args;
				break;
			}
			case T_OpExpr:
			{
				WhereClause = lappend(WhereClause, global_query->jointree->quals);
				break;
			}
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

//Planner
static PlannedStmt* QSOptimizer(Query* global_query, bool* graph, Index* transfer_array, int length)
{
	PlannedStmt* result = NULL;
	Query* best_query = NULL;
	//start_xact_command();
	int remain = hasNext(graph, length);
	int X = 0, Y = 0;
	Index rels[2] = { 0, 0 };
	if(query_splitting_algorithm == RelationshipCenter || query_splitting_algorithm == EntityCenter)
	{
		if (order_decision == global_view)
		{
			PlannedStmt* temp = neurqo_plan(copyObjectImpl(global_query), CURSOR_OPT_PARALLEL_OK, false);
			int leaf_has = 0, depth = 0;
			Plan* temp_plan = find_node_with_nleaf_recursive(temp->planTree, 2, &leaf_has, &depth);
			walk_plantree(temp_plan, rels);
			rels[0] = ((RangeTblEntry*)list_nth(global_query->rtable, rels[0] - 1))->relid;
			rels[1] = ((RangeTblEntry*)list_nth(global_query->rtable, rels[1] - 1))->relid;
		}
		for (int i = 0; i < length; i++)
		{
			if (remain > 1)
				mydest = DestIntoRel;
			else if (remain == 1)
				mydest = DestRemote;
			else if (remain == 0)
				return NULL;
			for (int j = 0; j < length; j++)
				transfer_array[j] = 0;
			//Get the rang table list for this subgraph
			List* rtable = getRT_2(global_query->rtable, graph, length, i, transfer_array);
			//Can this subgraph make a join ?
			if (rtable->length < 2)
			{
				continue;
			}
			char* relname = NULL;
			//If so, create a subquery
			Query* local_query = createQuery(global_query, mydest, rtable, transfer_array, length);
			PlannedStmt* candidate_result = neurqo_plan(copyObjectImpl(local_query), CURSOR_OPT_PARALLEL_OK, false);
			if (tarfunc(rels, candidate_result, result) == NEWBETTER)
			{
				if(result)
					pfree(result);
				result = candidate_result;
				best_query = local_query;
				X = i;
			}
			else
			{
				pfree(candidate_result);
			}
			candidate_result = NULL;
		}
	}
	else if (query_splitting_algorithm == Minsubquery)
	{
		if (order_decision == global_view)
		{
			PlannedStmt* temp = neurqo_plan(copyObjectImpl(global_query), CURSOR_OPT_PARALLEL_OK, false);
			int leaf_has = 0, depth = 0;
			Plan* temp_plan = find_node_with_nleaf_recursive(temp->planTree, 2, &leaf_has, &depth);
			walk_plantree(temp_plan, rels);
			rels[0] = ((RangeTblEntry*)list_nth(global_query->rtable, rels[0] - 1))->relid;
			rels[1] = ((RangeTblEntry*)list_nth(global_query->rtable, rels[1] - 1))->relid;
		}
		for (int i = 0; i < length; i++)
		{
			for (int j = i + 1; j < length; j++)
			{
				if (remain > 1)
					mydest = DestIntoRel;
				else if (remain == 1)
					mydest = DestRemote;
				else if (remain == 0)
					return NULL;
				for (int j = 0; j < length; j++)
					transfer_array[j] = 0;
				//Get the rang table list for this subgraph
				List* rtable = getRT_1(global_query->rtable, graph, length, i, j, transfer_array);
				//Can this subgraph make a join ?
				if (rtable == NIL)
				{
					continue;
				}
				char* relname = NULL;
				//If so, create a subquery
				Query* local_query = createQuery(global_query, mydest, rtable, transfer_array, length);
				PlannedStmt* candidate_result = neurqo_plan(copyObjectImpl(local_query), CURSOR_OPT_PARALLEL_OK, false);
				if (tarfunc(rels, candidate_result, result) == NEWBETTER)
				{
					if(result)
						pfree(result);
					result = candidate_result;
					best_query = local_query;
					X = i;
					Y = j;
				}
				else
				{
					pfree(candidate_result);
				}
				candidate_result = NULL;
			}
		}
	}
	for (int j = 0; j < length; j++)
		transfer_array[j] = 0;
	if (query_splitting_algorithm == RelationshipCenter || query_splitting_algorithm == EntityCenter)
	{
		Index index = 1;
		for (int j = 0; j < length; j++)
		{
			if (graph[X * length + j] == true)
			{
				transfer_array[j] = index++;
			}
			else if (X == j)
			{
				transfer_array[j] = index++;
			}
			else
			{
				transfer_array[j] = 0;
			}
		}
		for (int j = 0; j < length; j++)
		{
			if (graph[X * length + j] == true)
			{
				graph[X * length + j] = false;
				graph[j * length + X] = false;
			}
		}
	}
	else if (query_splitting_algorithm == Minsubquery)
	{
		for (int j = 0; j < length; j++)
			transfer_array[j] = 0;
		transfer_array[X] = 1;
		transfer_array[Y] = 2;
		graph[X * length + Y] = false;
		for (int i = 0; i < length; i++)
		{
			if (i < X)
			{
				if (graph[i * length + X] == true && graph[i * length + Y] == true)
				{
					graph[i * length + X] = false;
				}
			}
			else if (i > X && i < Y)
			{
				if (graph[X * length + i] == true && graph[i * length + Y] == true)
				{
					graph[X * length + i] = false;
				}
			}
			else if (i > Y)
			{
				if (graph[X * length + i] == true && graph[Y * length + i] == true)
				{
					graph[X * length + i] = false;
				}
			}
		}
	}
	switch (global_query->jointree->quals->type)
	{
		case T_BoolExpr:
		{
			((BoolExpr*)global_query->jointree->quals)->args = simplifyjoinlist(((BoolExpr*)global_query->jointree->quals)->args, mydest, transfer_array, graph, length);
			break;
		}
		case T_OpExpr:
		{
			global_query->jointree->quals = NULL;
			break;
		}
	}
	if (best_query != NULL)
	{
		if (result)
			pfree(result);
		result = neurqo_plan(best_query, CURSOR_OPT_PARALLEL_OK, true);
	}
	return result;
}

//Executor
static List* QSExecutor(char* query_string, CommandTag commandTag, Node* pstmt, PlannedStmt* plannedstmt, CommandDest dest, char* relname, QueryCompletion* completionTag, Query* querytree, Index* transfer_array, List* FKlist, MemoryContext oldcontext)
{
	Oid relid;
	int16 format;
	Portal portal;
	List* plantree_list;
	DestReceiver* receiver = NULL;
	bool is_parallel_worker = false;
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
	(void)PortalRun(portal, FETCH_ALL, true, true, receiver, receiver, completionTag);
	if (dest == DestIntoRel)
	{
		FKlist = Prepare4Next(querytree, transfer_array, (DR_intorel*)receiver, plannedstmt, relname, FKlist);
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
		if (transfer_array[var->varnosyn - 1] != 0)
		{
			RangeTblEntry* rte = (RangeTblEntry*)list_nth(global_query->rtable, var->varnosyn - 1);
			int len = strlen(rte->eref->aliasname) + strlen(strVal(list_nth(rte->eref->colnames, var->varattnosyn - 1))) + 2;
			char* attrname = (char*)palloc(len * sizeof(char));
			sprintf(attrname, "%s_%s", rte->eref->aliasname, strVal(list_nth(rte->eref->colnames, var->varattnosyn - 1)));
			var->varno = X + 1;
			var->varnosyn = var->varno;
			for (int i = 0; i < relation->rd_att->natts; i++)
			{
				if (strcmp(attrname, relation->rd_att->attrs[i].attname.data) == 0)
				{
					var->varattno = i + 1;
					var->varattnosyn = var->varattno;
					break;
				}
			}
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
		Var* vtar;
		if (tar->expr->type == T_Aggref)
		{
			TargetEntry* te = linitial(((Aggref*)tar->expr)->args);
			vtar = te->expr;
		}
		else
		{
			vtar = (Var*)tar->expr;
		}
		if (transfer_array[vtar->varnosyn - 1] != 0)
		{
			tar->resorigtbl = relid;
			for (int i = 0; i < relation->rd_att->natts; i++)
			{
				if (strcmp(tar->resname, relation->rd_att->attrs[i].attname.data) == 0)
				{
					tar->resorigcol = i + 1;
					vtar->varattno = i + 1;
					vtar->varno = X + 1;
					vtar->varattnosyn = vtar->varattno;
					vtar->varnosyn = vtar->varno;
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

//transfer joinlist to join graph
static bool* List2Graph(bool* is_relationship, List* joinlist, List* FKlist, int length)
{
	bool* graph = (bool*)palloc(length * length * sizeof(bool));
	memset(graph, false, length * length * sizeof(bool));
	ListCell* lc1;
	if (query_splitting_algorithm == RelationshipCenter || query_splitting_algorithm == EntityCenter)
	{
		foreach(lc1, FKlist)
		{
			bool flag = false;
			ForeignKeyOptInfo* fkOptInfo = (ForeignKeyOptInfo*)lfirst(lc1);
			int x = fkOptInfo->con_relid - 1;
			int y = fkOptInfo->ref_relid - 1;
			ListCell* lc2;
			foreach(lc2, joinlist)
			{
				Var* var1 = (Var*)linitial(((OpExpr*)lfirst(lc2))->args);
				Var* var2 = (Var*)lsecond(((OpExpr*)lfirst(lc2))->args);
				if (var1->varno - 1 == x && var2->varno - 1 == y)
				{
					flag = true;
					joinlist = foreach_delete_current(joinlist, lc2);
				}
				else if (var1->varno - 1 == y && var2->varno - 1 == x)
				{
					flag = true;
					joinlist = foreach_delete_current(joinlist, lc2);
				}
			}
			if (!flag)
				continue;
			if (query_splitting_algorithm == RelationshipCenter)
			{
				graph[x * length + y] = true;
			}
			else if (query_splitting_algorithm == EntityCenter)
			{
				graph[y * length + x] = true;
			}
		}
	}
	foreach(lc1, joinlist)
	{
		Var* var1 = (Var*)linitial(((OpExpr*)lfirst(lc1))->args);
		Var* var2 = (Var*)lsecond(((OpExpr*)lfirst(lc1))->args);
		if (query_splitting_algorithm == Minsubquery)
		{
			if (var1->varno > var2->varno)
			{
				graph[(var2->varno - 1) * length + var1->varno - 1] = true;
			}
			else if (var1->varno < var2->varno)
			{
				graph[(var1->varno - 1) * length + var2->varno - 1] = true;
			}
		}
		else if (query_splitting_algorithm == RelationshipCenter)
		{
			if ((is_relationship[var1->varno - 1] == true) && (is_relationship[var2->varno - 1] == false))
			{
				graph[(var1->varno - 1) * length + var2->varno - 1] = true;
			}
			else if ((is_relationship[var1->varno - 1] == false) && (is_relationship[var2->varno - 1] == true))
			{
				graph[(var2->varno - 1) * length + var1->varno - 1] = true;
			}
			else if ((is_relationship[var1->varno - 1] == false) && (is_relationship[var2->varno - 1] == false))
			{
				graph[(var1->varno - 1) * length + var2->varno - 1] = true;
				graph[(var2->varno - 1) * length + var1->varno - 1] = true;
			}
			else
			{
				graph[(var1->varno - 1) * length + var2->varno - 1] = true;
				graph[(var2->varno - 1) * length + var1->varno - 1] = true;
			}
		}
		else if(query_splitting_algorithm == EntityCenter)
		{
			if ((is_relationship[var1->varno - 1] == false) && (is_relationship[var2->varno - 1] == true))
			{
				graph[(var1->varno - 1) * length + var2->varno - 1] = true;
			}
			else if ((is_relationship[var1->varno - 1] == true) && (is_relationship[var2->varno - 1] == false))
			{
				graph[(var2->varno - 1) * length + var1->varno - 1] = true;
			}
			else if ((is_relationship[var1->varno - 1] == false) && (is_relationship[var2->varno - 1] == false))
			{
				graph[(var1->varno - 1) * length + var2->varno - 1] = true;
				graph[(var2->varno - 1) * length + var1->varno - 1] = true;
			}
			else
			{
				graph[(var1->varno - 1) * length + var2->varno - 1] = true;
				graph[(var2->varno - 1) * length + var1->varno - 1] = true;
			}
		}
	}
	return graph;
}

static bool is_ER(OpExpr* opexpr, bool* is_relationship, int length)
{
	Var* var1 = (Var*)linitial(opexpr->args);
	Var* var2 = (Var*)lsecond(opexpr->args);
	if (is_relationship[var1->varno - 1] && is_relationship[var2->varno - 1])
		return false;
	else if (is_relationship[var1->varno - 1] || is_relationship[var2->varno - 1])
		return true;
	else
		return false;
}

static bool is_FK(OpExpr* opexpr, List* fklist)
{
	ListCell* lc = NULL;
	Var* var1 = (Var*)linitial(opexpr->args);
	Var* var2 = (Var*)lsecond(opexpr->args);
	foreach(lc, fklist)
	{
		ForeignKeyOptInfo* fkOptInfo = (ForeignKeyOptInfo*)lfirst(lc);
		if (var1->varno == fkOptInfo->con_relid && var2->varno == fkOptInfo->ref_relid)
			return true;
		else if (var2->varno == fkOptInfo->con_relid && var1->varno == fkOptInfo->ref_relid)
			return true;
	}
	return false;
}

static bool is_2relationship(OpExpr* opexpr, bool* is_relationship, int length)
{
	Var* var1 = (Var*)linitial(opexpr->args);
	Var* var2 = (Var*)lsecond(opexpr->args);
	if (is_relationship[var1->varno - 1] && is_relationship[var2->varno - 1])
		return true;
	return false;
}

//Expr is a filter clause?
static bool is_RC(Expr* expr)
{
	if (expr->type != T_OpExpr)
		return true;
	OpExpr* opexpr = (OpExpr*)expr;
	return (((Node*)lsecond(opexpr->args))->type == T_Const);
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
		if (!is_RC(expr))
		{
			OpExpr* opexpr = (OpExpr*)expr;
			NodeTag type = ((Node*)linitial(opexpr->args))->type;
			Var* var1 = linitial(opexpr->args);
			Var* var2 = (Var*)lsecond(opexpr->args);
			//µ±Ç°queryµ½ÍâÎ§
			if (transfer_array[var1->varno - 1] != 0 && transfer_array[var2->varno - 1] == 0)
			{
				ListCell* lc1;
				bool append = true;
				foreach(lc1, reslist)
				{
					Var* var = (Var*)lfirst(lc1);
					if (var->varattno == var1->varattno && var->varno == var1->varno)
					{
						append = false;
						break;
					}
				}
				if (append)
				{
					Var* var = copyObjectImpl(var1);
					reslist = lappend(reslist, var);
				}
			}
			else if (transfer_array[var1->varno - 1] == 0 && transfer_array[var2->varno - 1] != 0)
			{
				ListCell* lc1;
				bool append = true;
				foreach(lc1, reslist)
				{
					Var* var = (Var*)lfirst(lc1);
					if (var->varattno == var2->varattno && var->varno == var2->varno)
					{
						append = false;
						break;
					}
				}
				if (append)
				{
					Var* var = copyObjectImpl(var2);
					reslist = lappend(reslist, var);
				}
			}
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
	switch (query->jointree->quals->type)
	{
		case T_BoolExpr:
		{
			varlist = findvarlist(((BoolExpr*)query->jointree->quals)->args, transfer_array, length);
			break;
		}
		case T_OpExpr:
		{
			List* temp = lappend(NIL, query->jointree->quals);
			varlist = findvarlist(temp, transfer_array, length);
			break;
		}
	}
	query->targetList = settargetlist(global_query->rtable, rtable, dest, varlist, query->targetList, transfer_array, length);
	switch (query->jointree->quals->type)
	{
		case T_BoolExpr:
		{
			((BoolExpr*)query->jointree->quals)->args = setjoinlist(((BoolExpr*)query->jointree->quals)->args, dest, transfer_array, length);
			break;
		}
		case T_OpExpr:
			break;
	}
	if (dest == DestRemote)
	{
		query->hasAggs = global_query->hasAggs;
	}
	else
	{
		query->hasAggs = false;
	}
	return query;
}

static bool doNullTestTransfor(NullTest* expr, Index* transfer_array)
{
	NodeTag type = nodeTag(expr->arg);
	Var* var = NULL;
	if (type == T_Var)
		var = (Var*)expr->arg;
	else if (type == T_RelabelType)
		var = (Var*)((RelabelType*)expr->arg)->arg;
	if (transfer_array[var->varno - 1] == 0)
		return false;
	var->varno = transfer_array[var->varno - 1];
	var->varnosyn = var->varno;
	return true;
}

static bool doOpExprTransfor(OpExpr* expr, Index* transfer_array)
{
	bool flag = true;
	NodeTag type1 = ((Node*)linitial(expr->args))->type;
	Var* var1 = NULL;
	Var* var2 = NULL;
	if (type1 == T_Var)
	{
		var1 = (Var*)linitial(expr->args);
	}
	else if (type1 == T_RelabelType)
	{
		var1 = (Var*)((RelabelType*)linitial(expr->args))->arg;
	}
	NodeTag type2 = ((Node*)lsecond(expr->args))->type;
	if (type2 == T_Var)
	{
		var2 = (Var*)lsecond(expr->args);
	}
	else if (type2 == T_RelabelType)
	{
		var2 = (Var*)((RelabelType*)lsecond(expr->args))->arg;
	}
	if (var1 && transfer_array[var1->varno - 1] == 0)
	{
		flag = false;
	}
	else if (var1)
	{
		var1->varno = transfer_array[var1->varno - 1];
		var1->varnosyn = var1->varno;
	}
	if (var2 && transfer_array[var2->varno - 1] == 0)
	{
		flag = false;
	}
	else if (var2)
	{
		var2->varno = transfer_array[var2->varno - 1];
		var2->varnosyn = var2->varno;
	}
	return flag;
}

static bool doScalarArrayOpExprTransfor(ScalarArrayOpExpr* expr, Index* transfer_array)
{
	NodeTag type = nodeTag(linitial(expr->args));
	Var* var = NULL;
	if (type == T_Var)
		var = (Var*)linitial(expr->args);
	else if (type == T_RelabelType)
		var = (Var*)((RelabelType*)linitial(expr->args))->arg;
	if (transfer_array[var->varno - 1] == 0)
		return false;
	var->varno = transfer_array[var->varno - 1];
	var->varnosyn = var->varno;
	return true;
}

static List* setjoinlist(List* qualslist, CommandDest dest, Index* transfer_array, int length)
{
	ListCell* lc;
	foreach(lc, qualslist)
	{
		Expr* expr = (Expr*)lfirst(lc);
		bool flag = true;
		switch (expr->type)
		{
			case T_NullTest:
			{
				NullTest* nulltest = (NullTest*)expr;
				flag = doNullTestTransfor(nulltest, transfer_array);
				break;
			}
			case T_OpExpr:
			{
				OpExpr* opexpr = (OpExpr*)expr;
				flag = doOpExprTransfor(opexpr, transfer_array);
				break;
			}
			case T_ScalarArrayOpExpr:
			{
				ScalarArrayOpExpr* scalararrayopexpr = (ScalarArrayOpExpr*)expr;
				flag = doScalarArrayOpExprTransfor(scalararrayopexpr, transfer_array);
				break;
			}
			case T_BoolExpr:
			{
				BoolExpr* boolexpr = (BoolExpr*)expr;
				boolexpr->args = setjoinlist(boolexpr->args, dest, transfer_array, length);
				if (boolexpr->args == NULL)
					flag = false;
				break;
			}
		}
		if (!flag)
		{
			qualslist = foreach_delete_current(qualslist, lc);
		}
	}
	return qualslist;
}

static List* simplifyjoinlist(List* list, CommandDest dest, Index* transfer_array, bool* graph, int length)
{
	ListCell* lc;
	foreach(lc, list)
	{
		Expr* expr = (Expr*)lfirst(lc);
		if (is_RC(expr))
		{
			List* varlist = pull_var_clause(expr, 0);
			Var* var = (Var*)linitial(varlist);
			if (transfer_array[var->varno - 1] != 0)
				list = foreach_delete_current(list, lc);
		}
		else
		{
			bool flag = false;
			Index X = 0, Y = 0;
			OpExpr* opexpr = (OpExpr*)expr;
			Var* var1 = (Var*)linitial(opexpr->args);
			Var* var2 = (Var*)lsecond(opexpr->args);
			Assert(var1->varno != var2->varno);
			X = var1->varno - 1;
			Y = var2->varno - 1;
			if (graph[X * length + Y] == false && graph[Y * length + X] == false)
			{
				list = foreach_delete_current(list, lc);
			}
		}
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
		Var* vtar;
		if (tar->expr->type != T_Var)
		{
			continue;
		}
		vtar = (Var*)tar->expr;
		if (vtar->varlevelsup == 0 &&
			vtar->varno > 0 &&
			vtar->varno <= length &&
			transfer_array[vtar->varno - 1] != 0)
		{
			vtar->varno = transfer_array[vtar->varno - 1];
			vtar->varnosyn = vtar->varno;
			continue;
		}
		targetlist = foreach_delete_current(targetlist, lc);
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
			TargetEntry* tar = linitial(((Aggref*)old_tar->expr)->args);
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
