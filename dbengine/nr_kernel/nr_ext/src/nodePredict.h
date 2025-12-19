#ifndef NODENEURDBPREDICT_H
#define NODENEURDBPREDICT_H

#include "nodes/execnodes.h"

extern NeurDBPredictState * ExecInitNeurDBPredict(NeurDBPredict * node, EState *estate, int eflags);
extern void ExecEndNeurDBPredict(NeurDBPredictState * node);
extern void ExecReScanNeurDBPredict(NeurDBPredictState * node);

#endif							/* NODENEURDBPREDICT_H */
