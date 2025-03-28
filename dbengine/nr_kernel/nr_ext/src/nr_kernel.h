#ifndef NR_KERNEL_H
#define NR_KERNEL_H

#include "postgres.h"

#include "executor/executor.h"

extern PlanState *NeurDB_ExecInitNode(Plan *node, EState *estate, int eflags);
extern PlanState *NeurDB_ExecEndNode(PlanState *node);

#endif
