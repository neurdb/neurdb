#ifndef PREDICT_H
#define PREDICT_H

#include "catalog/objectaddress.h"
#include "nodes/parsenodes.h"
#include "parser/parse_node.h"
#include "tcop/dest.h"

/*
 * GUC variable for current configuration
 */
extern PGDLLIMPORT char *NrModelName;
extern PGDLLIMPORT int NrTaskBatchSize;
extern PGDLLIMPORT int NrTaskEpoch;
extern PGDLLIMPORT int NrTaskMaxFeatures;
extern PGDLLIMPORT int NrTaskNumBatches;

extern ObjectAddress ExecPredictStmt(NeurDBPredictStmt * stmt, ParseState *pstate, const char *whereClauseString, DestReceiver *dest);
#endif							/* PREDICT_H */
