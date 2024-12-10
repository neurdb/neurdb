/*-------------------------------------------------------------------------
 *
 * predict.h
 *	  prototypes for commands/predict.c
 *
 *
 * Portions Copyright (c) 1996-2023, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * src/include/commands/predict.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef PREDICT_H
#define PREDICT_H

#include "catalog/objectaddress.h"
#include "nodes/parsenodes.h"
#include "parser/parse_node.h"

/*
 * GUC variable for current configuration
 */
extern PGDLLIMPORT char *NRModelName;
extern PGDLLIMPORT int NRTaskBatchSize;
extern PGDLLIMPORT int NRTaskEpoch;
extern PGDLLIMPORT int NRTaskMaxFeatures;
extern PGDLLIMPORT int NRTaskNumBatches;

extern void exec_udf(const char *model, const char *table, const char *trainColumns, const char *targetColumn, const char *whereClause);
extern ObjectAddress ExecPredictStmt(NeurDBPredictStmt * stmt, ParseState *pstate, const char *whereClauseString);
#endif							/* PREDICT_H */
