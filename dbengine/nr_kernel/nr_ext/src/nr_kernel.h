#ifndef NR_KERNEL_H
#define NR_KERNEL_H

#include "postgres.h"

#include "optimizer/planner.h"
#include "executor/executor.h"


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
