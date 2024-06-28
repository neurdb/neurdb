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
#include "nodes/primnodes.h"

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
    FmgrInfo findModelFmgrInfo;
    FmgrInfo trainingFmgrInfo;
    FmgrInfo inferenceFmgrInfo;

    LOCAL_FCINFO(findModelFCInfo, FUNC_MAX_ARGS); // fcinfo for findModel function
    LOCAL_FCINFO(trainingFCInfo, FUNC_MAX_ARGS); // fcinfo for training function
    LOCAL_FCINFO(inferenceFCInfo, FUNC_MAX_ARGS); // fcinfo for inference function

    char modelName[] = "MLP";   // name of the model
    Datum findModelResult;
    Datum trainingResult;
    Datum inferenceResult;
    Oid findModelArgTypes[1] = {TEXTOID};
    Oid trainingArgTypes[4] = {TEXTOID, TEXTOID, TEXTOID, TEXTOID};
    Oid inferenceArgTypes[4] = {TEXTOID, INT4OID, TEXTOID, TEXTARRAYOID};
    char findModelFuncName[] = "pgm_get_model_id_by_name";
    char trainingFuncName[] = "mlp_clf";
    char inferenceFuncName[] = "pgm_predict_table";

    if (columns == NULL || table == NULL || whereClause == NULL) {
        elog(ERROR, "Null argument passed to exec_udf");
        return;
    }

    // check if the model exists
    Oid findModelFuncOid = LookupFuncName(list_make1(makeString(findModelFuncName)), 1, findModelArgTypes, false);

    if (!OidIsValid(findModelFuncOid)) {
        elog(ERROR, "Function %s not found", findModelFuncName);
        return;
    }

    // get the model id
    fmgr_info(findModelFuncOid, &findModelFmgrInfo);
    InitFunctionCallInfoData(*findModelFCInfo, &findModelFmgrInfo, 1, InvalidOid, NULL, NULL);

    findModelFCInfo->args[0].value = CStringGetTextDatum(modelName);
    findModelFCInfo->args[0].isnull = false;
    findModelResult = FunctionCallInvoke(findModelFCInfo);

    if (findModelResult == 0) {
        // model not found, train the model
        elog(INFO, "Model not found, training model");

        Oid trainingFuncOid = LookupFuncName(list_make1(makeString(trainingFuncName)), 4, trainingArgTypes, false);
        if (!OidIsValid(trainingFuncOid)) {
            elog(ERROR, "Function %s not found", trainingFuncName);
            return;
        }

        fmgr_info(trainingFuncOid, &trainingFmgrInfo);
        InitFunctionCallInfoData(*trainingFCInfo, &trainingFmgrInfo, 4, InvalidOid, NULL, NULL);

        trainingFCInfo->args[0].value = CStringGetTextDatum(columns);
        trainingFCInfo->args[1].value = CStringGetTextDatum(table);
        trainingFCInfo->args[2].value = CStringGetTextDatum(whereClause);
        trainingFCInfo->args[3].value = CStringGetTextDatum("/code/neurdb-dev/contrib/nr/pysrc/config.ini");

        trainingFCInfo->args[0].isnull = false;
        trainingFCInfo->args[1].isnull = false;
        trainingFCInfo->args[2].isnull = false;
        trainingFCInfo->args[3].isnull = false;

        trainingResult = FunctionCallInvoke(trainingFCInfo);

        if (!trainingFCInfo->isnull) {
            text *resultText = DatumGetTextP(trainingResult);
            char *resultCString = text_to_cstring(resultText);
            pfree(resultCString);
        } else {
            elog(INFO, "Result is NULL");
        }
    } else {
        // model found, make a prediction
        elog(INFO, "Model found, making prediction");

        Oid inferenceFuncOid = LookupFuncName(list_make1(makeString(inferenceFuncName)), 4, inferenceArgTypes, false);
        if (!OidIsValid(inferenceFuncOid)) {
            elog(ERROR, "Function %s not found", inferenceFuncName);
            return;
        }

        fmgr_info(inferenceFuncOid, &inferenceFmgrInfo);
        InitFunctionCallInfoData(*inferenceFCInfo, &inferenceFmgrInfo, 4, InvalidOid, NULL, NULL);

        inferenceFCInfo->args[0].value = CStringGetTextDatum(modelName);
        inferenceFCInfo->args[1].value = Int32GetDatum(100);        // change inference batch size here
        inferenceFCInfo->args[2].value = CStringGetTextDatum(table);
        inferenceFCInfo->args[3].value = CStringGetTextDatum(columns);

        inferenceFCInfo->args[0].isnull = false;
        inferenceFCInfo->args[1].isnull = false;
        inferenceFCInfo->args[2].isnull = false;
        inferenceFCInfo->args[3].isnull = false;

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
    char *whereClause = NULL;

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
    if (stmt->whereClause != NULL) {
        whereClause = nodeToString(stmt->whereClause);
        elog(DEBUG1, "Extracted where clause: %s", whereClause);
    } else {
        elog(DEBUG1, "No where clause provided");
        whereClause = "";
    }

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
