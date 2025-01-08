#ifndef NR_KERNEL_H
#define NR_KERNEL_H

#include "postgres.h"

#include "executor/executor.h"

extern PlanState *NeurDB_ExecInitNode(Plan *node, EState *estate, int eflags);

#endif
