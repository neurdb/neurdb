/*-------------------------------------------------------------------------
 *
 * predict.c
 *	  Commands for NeurDB predict commands.
 *
 * Copyright (c) 2024, NeurDB Contributors
 *
 * src/backend/commands/predict.c
 *
 *-------------------------------------------------------------------------
 */
#include "postgres.h"

#include "commands/predict.h"
#include "parser/parse_func.h"
#include "parser/parse_node.h"
#include "parser/parse_target.h"
#include "catalog/pg_type.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "nodes/primnodes.h"


static List *
split_columns(const char *columns) {
    List *result = NIL;
    char *token = strtok(columns, ",");
    while (token != NULL) {
        result = lappend(result, makeString(token));
        token = strtok(NULL, ",");
    }
    return result;
}

/*
* exec_udf --- Execute customer UDFs
*
* columns: columns names.
* table: table used.
* whereClause: where condition.
*
* @note: parameters in exec_udf are all hard-coded for now. In the future, we
* will need to change this to adapt to the actual situation. This function
* needs to be revamped to be more general.
*/
void
exec_udf(const char *columns, const char *table, const char *whereClause) {
    // initialize function call infos
    FmgrInfo inferenceFmgrInfo;
    LOCAL_FCINFO(inferenceFCInfo, FUNC_MAX_ARGS); // fcinfo for inference function

    // TODO: These parameters should be passed in as arguments, hard-coded for now
    char modelName[] = "armnet"; // name of the model
    int modelId = 1;
    int batch_size = 4096;
    int batch_num = 80;
    int nfeat = 10;

    Datum inferenceResult;
    Oid inferenceArgTypes[7] = {TEXTOID, INT4OID, TEXTOID, INT4OID, INT4OID, INT4OID, TEXTARRAYOID};
    char inferenceFuncName[] = "nr_inference";

    if (columns == NULL || table == NULL || whereClause == NULL) {
        elog(ERROR, "Null argument passed to exec_udf");
        return;
    }

    Oid inferenceFuncOid = LookupFuncName(list_make1(makeString(inferenceFuncName)), 7, inferenceArgTypes, false);
    if (!OidIsValid(inferenceFuncOid)) {
        elog(ERROR, "Function %s not found", inferenceFuncName);
        return;
    }

    fmgr_info(inferenceFuncOid, &inferenceFmgrInfo);
    InitFunctionCallInfoData(*inferenceFCInfo, &inferenceFmgrInfo, 4, InvalidOid, NULL, NULL);

    // split columns into an array of text
    List *columnList = split_columns(columns);
    int ncolumns = list_length(columnList);

    Datum *columnDatums = (Datum *) palloc(sizeof(Datum) * ncolumns);
    for (int i = 0; i < ncolumns; i++) {
        columnDatums[i] = CStringGetTextDatum(strVal(list_nth(columnList, i)));
    }

    ArrayType *columnArray = construct_array(columnDatums, ncolumns, TEXTOID, -1, false, 'i');

    inferenceFCInfo->args[0].value = CStringGetTextDatum(modelName);
    inferenceFCInfo->args[1].value = Int32GetDatum(modelId);
    inferenceFCInfo->args[2].value = CStringGetTextDatum(table);
    inferenceFCInfo->args[3].value = Int32GetDatum(batch_size);
    inferenceFCInfo->args[4].value = Int32GetDatum(batch_num);
    inferenceFCInfo->args[5].value = Int32GetDatum(nfeat);
    inferenceFCInfo->args[6].value = PointerGetDatum(columnArray);

    inferenceFCInfo->args[0].isnull = false;
    inferenceFCInfo->args[1].isnull = false;
    inferenceFCInfo->args[2].isnull = false;
    inferenceFCInfo->args[3].isnull = false;
    inferenceFCInfo->args[4].isnull = false;
    inferenceFCInfo->args[5].isnull = false;
    inferenceFCInfo->args[6].isnull = false;

    inferenceResult = FunctionCallInvoke(inferenceFCInfo);
    if (!inferenceFCInfo->isnull) {
        text *resultText = DatumGetTextP(inferenceResult);
        char *resultCString = text_to_cstring(resultText);
        elog(INFO, "Inference result: %s", resultCString);
        pfree(resultCString);
    } else {
        elog(INFO, "Inference result is NULL");
    }
}

/*
* ExecPredictStmt --- Execution for node NeurDBPredictStmt defined in include/nodes/parsenodes.h
*
* NeurDBPredictStmt: Node structure in include/nodes/parsenodes.h
*/
ObjectAddress
ExecPredictStmt(NeurDBPredictStmt *stmt, ParseState *pstate, const char *whereClauseString) {
    List *p_target = NIL; /* A list of targets (columns) for the prediction */
    ListCell *o_target;
    StringInfoData columns;
    char *tableName = NULL;
    char *whereClause = "<DEPRECATED>";

    elog(DEBUG1, "Starting ExecPredictStmt");
    initStringInfo(&columns);

    /* Extract the column names from targetList and combine them into a single string */
    foreach(o_target, stmt->targetList) {
        ResTarget *res = (ResTarget *) lfirst(o_target);

        if (res == NULL || res->val == NULL) {
            elog(ERROR, "Null target column in statement");
            return InvalidObjectAddress;
        }
        char *colname = FigureColname(res->val);

        if (colname == NULL) {
            elog(ERROR, "Null column name in target list");
            return InvalidObjectAddress;
        }
        appendStringInfo(&columns, "%s,", colname);
    }

    if (columns.len > 0) {
        //Remove the trailing comma
        columns.data[columns.len - 1] = '\0';
    }

    elog(DEBUG1, "Collected columns: %s", columns.data);

    /* Extract the table name from fromClause */
    if (stmt->fromClause != NIL) {
        RangeVar *rv = (RangeVar *) linitial(stmt->fromClause);

        if (rv == NULL) {
            elog(ERROR, "Null range variable in from clause");
            return InvalidObjectAddress;
        }
        tableName = rv->relname;
        elog(DEBUG1, "Extracted table name: %s", tableName);
    } else {
        elog(ERROR, "No from clause in statement");
        return InvalidObjectAddress;
    }

    /* Convert whereClause to string */
    // if (stmt->whereClause != NULL) {
    //     whereClause = nodeToString(stmt->whereClause);
    //     elog(DEBUG1, "Extracted where clause: %s", whereClause);
    // } else {
    //     elog(DEBUG1, "No where clause provided");
    //     whereClause = "";
    // }

    /* Execute the UDF with extracted columns, table name, and where clause */
    elog(DEBUG1, "Executing UDF with columns: %s, table: %s, whereClause: %s", columns.data, tableName, whereClause);
    exec_udf(columns.data, tableName, whereClause);

    /* Free the list of targets */
    list_free(p_target);

    return InvalidObjectAddress;

    /**
     * TODO:
     * It looks like calling stmt-> kind will cause memory leak. So I commented
     * it out for now. We will solve this issue later.
     */
    // switch (stmt->kind)
    // {
    // 	case PREDICT_CLASS:
    // 		elog(ERROR, "PREDICT_CLASS prediction type not implemented: %d", (int) stmt->kind);
    // 		return InvalidObjectAddress;
    //
    // 	case PREDICT_VALUE:
    // 		elog(ERROR, "PREDICT_VALUE prediction type not implemented: %d", (int) stmt->kind);
    // 		return InvalidObjectAddress;
    //
    // 	default:
    // 		elog(ERROR, "unrecognized prediction type: %d", (int) stmt->kind);
    // 		return InvalidObjectAddress;
    // }
}
