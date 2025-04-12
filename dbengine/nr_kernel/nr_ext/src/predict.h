#ifndef PREDICT_H
#define PREDICT_H

#include "catalog/objectaddress.h"
#include "neurdb.h"
#include "nodes/parsenodes.h"
#include "parser/parse_node.h"
#include "tcop/dest.h"

extern ObjectAddress ExecPredictStmt(NeurDBPredictStmt * stmt, ParseState *pstate, const char *whereClauseString, DestReceiver *dest);
#endif
