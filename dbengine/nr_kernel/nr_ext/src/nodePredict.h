#ifndef NODENEURDBPREDICT_H
#define NODENEURDBPREDICT_H

#include "nodes/execnodes.h"

extern NeurDBPredictState * ExecInitNeurDBPredict(NeurDBPredict * node, EState *estate, int eflags);
extern void ExecEndNeurDBPredict(NeurDBPredictState * node);
extern void ExecReScanNeurDBPredict(NeurDBPredictState * node);

/*
 * Wrappers with the generic signatures expected by the core executor's
 * NeurDBPredict dispatch hooks (see executor/executor.h), so NeurDBPredict
 * nodes nested inside plan trees (under SubqueryScan etc.) can be executed.
 */
extern PlanState *NeurDBPredictInitNodeHook(Plan *node, EState *estate, int eflags);
extern void NeurDBPredictEndNodeHook(PlanState *node);
extern void NeurDBPredictReScanHook(PlanState *node);

#endif							/* NODENEURDBPREDICT_H */
