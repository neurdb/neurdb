#ifndef PREDICT_H
#define PREDICT_H

#include "catalog/objectaddress.h"
#include "nodes/parsenodes.h"
#include "parser/parse_node.h"
#include "tcop/dest.h"

/*
 * GUC variable for current configuration
 */
extern PGDLLIMPORT char *NRModelName;
extern PGDLLIMPORT int NRTaskBatchSize;
extern PGDLLIMPORT int NRTaskEpoch;
extern PGDLLIMPORT int NRTaskMaxFeatures;
extern PGDLLIMPORT int NRTaskNumBatches;

extern ObjectAddress ExecPredictStmt(NeurDBPredictStmt * stmt, ParseState *pstate, const char *whereClauseString, DestReceiver *dest);
#endif							/* PREDICT_H */
