#ifndef NR_KERNEL_H
#define NR_KERNEL_H

#include "postgres.h"

#include "nodes/pathnodes.h"
#include "nodes/plannodes.h"

#include "executor/executor.h"

/* Hook for plugins to get control in planner() */
typedef PlannedStmt *(*planner_hook_type) (Query *parse,
										   const char *query_string,
										   int cursorOptions,
										   ParamListInfo boundParams);
extern PGDLLIMPORT planner_hook_type planner_hook;

/* Hook for plugins to get control when grouping_planner() plans upper rels */
typedef void (*create_upper_paths_hook_type) (PlannerInfo *root,
											  UpperRelationKind stage,
											  RelOptInfo *input_rel,
											  RelOptInfo *output_rel,
											  void *extra);
extern PGDLLIMPORT create_upper_paths_hook_type create_upper_paths_hook;


extern PlanState *NeurDB_ExecInitNode(Plan *node, EState *estate, int eflags);
extern PlanState *NeurDB_ExecEndNode(PlanState *node);


extern planner_hook_type original_planner_hook;

extern ExecutorStart_hook_type original_executorstart_hook;
extern ExecutorRun_hook_type original_executorrun_hook;
extern ExecutorEnd_hook_type original_executorend_hook;
extern ExecutorFinish_hook_type original_executorfinish_hook;


PlannedStmt *NeurDB_planner(Query *parse, const char *query_string, int cursorOptions, ParamListInfo boundParams);

void NeurDB_ExecutorStart(QueryDesc *queryDesc, int eflags);
void NeurDB_ExecutorRun(QueryDesc *queryDesc, ScanDirection direction, uint64 count, bool execute_once);
void NeurDB_ExecutorEnd(QueryDesc *queryDesc);
void NeurDB_ExecutorFinish(QueryDesc *queryDesc);

#endif
