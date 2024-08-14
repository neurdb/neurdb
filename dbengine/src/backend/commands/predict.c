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
exec_udf(const char *table, const char *trainColumns, const char *targetColumn, const char *whereClause) {
    // lookup function call infos
    FmgrInfo modelLookupFmgrInfo;
    LOCAL_FCINFO(modelLookupFCInfo, FUNC_MAX_ARGS); // fcinfo for lookup function
    Datum modelLookupResult;
    Oid modelLookupArgTypes[3] = {TEXTOID, TEXTARRAYOID, TEXTOID};
    char modelLookupFuncName[] = "nr_model_lookup";

    Oid modelLookupFuncOid = LookupFuncName(list_make1(makeString(modelLookupFuncName)), 3, modelLookupArgTypes, false);
    if (!OidIsValid(modelLookupFuncOid)) {
        elog(ERROR, "Function %s not found", modelLookupFuncName);
        return;
    }

    fmgr_info(modelLookupFuncOid, &modelLookupFmgrInfo);
    InitFunctionCallInfoData(*modelLookupFCInfo, &modelLookupFmgrInfo, 3, InvalidOid, NULL, NULL);

    // split columns into an array of text
    List *trainColumnsList = split_columns(trainColumns);
    int nTrainColumns = list_length(trainColumnsList);

    Datum *trainColumnDatums = (Datum *) palloc(sizeof(Datum) * nTrainColumns);
    for (int i = 0; i < nTrainColumns; i++) {
        trainColumnDatums[i] = CStringGetTextDatum(strVal(list_nth(trainColumnsList, i)));
    }
    ArrayType *trainColumnArray = construct_array(trainColumnDatums, nTrainColumns, TEXTOID, -1, false, 'i');

    modelLookupFCInfo->args[0].value = CStringGetTextDatum(table);
    modelLookupFCInfo->args[1].value = PointerGetDatum(trainColumnArray);
    modelLookupFCInfo->args[2].value = CStringGetTextDatum(targetColumn);

    modelLookupResult = FunctionCallInvoke(modelLookupFCInfo);
    int modelId = 0;
    // modelLookupResult is a boolean
    if (!modelLookupFCInfo->isnull) {
        modelId = DatumGetInt32(modelLookupResult);
    }

    if (modelId == 0) {
        // model does not exist, training
        FmgrInfo trainingFmgrInfo;
        LOCAL_FCINFO(trainingFCInfo, FUNC_MAX_ARGS); // fcinfo for training function

        // TODO: These parameters should be passed in as arguments, hard-coded for now
        char modelName[] = "armnet";
        int batch_size = 4096;
        int batch_num = 80;
        int nfeat = 10;
        int epoch = 1;

        Datum trainingResult;

        Oid trainingArgTypes[8] = {TEXTOID, TEXTOID, INT4OID, INT4OID, INT4OID, INT4OID, TEXTARRAYOID, TEXTOID};
        char trainingFuncName[] = "nr_train";

        Oid trainingFuncOid = LookupFuncName(list_make1(makeString(trainingFuncName)), 8, trainingArgTypes, false);
        if (!OidIsValid(trainingFuncOid)) {
            elog(ERROR, "Function %s not found", trainingFuncName);
            return;
        }

        fmgr_info(trainingFuncOid, &trainingFmgrInfo);
        InitFunctionCallInfoData(*trainingFCInfo, &trainingFmgrInfo, 8, InvalidOid, NULL, NULL);

        trainingFCInfo->args[0].value = CStringGetTextDatum(modelName);
        trainingFCInfo->args[1].value = CStringGetTextDatum(table);
        trainingFCInfo->args[2].value = Int32GetDatum(batch_size);
        trainingFCInfo->args[3].value = Int32GetDatum(batch_num);
        trainingFCInfo->args[4].value = Int32GetDatum(epoch);
        trainingFCInfo->args[5].value = Int32GetDatum(nfeat);
        trainingFCInfo->args[6].value = PointerGetDatum(trainColumnArray);
        trainingFCInfo->args[7].value = CStringGetTextDatum(targetColumn);

        trainingFCInfo->args[0].isnull = false;
        trainingFCInfo->args[1].isnull = false;
        trainingFCInfo->args[2].isnull = false;
        trainingFCInfo->args[3].isnull = false;
        trainingFCInfo->args[4].isnull = false;
        trainingFCInfo->args[5].isnull = false;
        trainingFCInfo->args[6].isnull = false;
        trainingFCInfo->args[7].isnull = false;

        trainingResult = FunctionCallInvoke(trainingFCInfo);
        if (!trainingFCInfo->isnull) {
            text *resultText = DatumGetTextP(trainingResult);
            char *resultCString = text_to_cstring(resultText);
            elog(INFO, "Training result: %s", resultCString);
            pfree(resultCString);
        } else {
            elog(INFO, "Training result is NULL");
        }
    } else {
        // inference
        FmgrInfo inferenceFmgrInfo;
        LOCAL_FCINFO(inferenceFCInfo, FUNC_MAX_ARGS); // fcinfo for inference function

        // TODO: These parameters should be passed in as arguments, hard-coded for now
        char modelName[] = "armnet";
        int batch_size = 4096;
        int batch_num = 80;
        int nfeat = 10;

        Datum inferenceResult;
        Oid inferenceArgTypes[7] = {TEXTOID, INT4OID, TEXTOID, INT4OID, INT4OID, INT4OID, TEXTARRAYOID};
        char inferenceFuncName[] = "nr_inference";

        Oid inferenceFuncOid = LookupFuncName(list_make1(makeString(inferenceFuncName)), 7, inferenceArgTypes, false);
        if (!OidIsValid(inferenceFuncOid)) {
            elog(ERROR, "Function %s not found", inferenceFuncName);
            return;
        }

        fmgr_info(inferenceFuncOid, &inferenceFmgrInfo);
        InitFunctionCallInfoData(*inferenceFCInfo, &inferenceFmgrInfo, 4, InvalidOid, NULL, NULL);

        inferenceFCInfo->args[0].value = CStringGetTextDatum(modelName);
        inferenceFCInfo->args[1].value = Int32GetDatum(modelId);
        inferenceFCInfo->args[2].value = CStringGetTextDatum(table);
        inferenceFCInfo->args[3].value = Int32GetDatum(batch_size);
        inferenceFCInfo->args[4].value = Int32GetDatum(batch_num);
        inferenceFCInfo->args[5].value = Int32GetDatum(nfeat);
        inferenceFCInfo->args[6].value = PointerGetDatum(trainColumnArray);

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
}

/*
* ExecPredictStmt --- Execution for node NeurDBPredictStmt defined in include/nodes/parsenodes.h
*
* NeurDBPredictStmt: Node structure in include/nodes/parsenodes.h
*/
ObjectAddress
ExecPredictStmt(NeurDBPredictStmt *stmt, ParseState *pstate, const char *whereClauseString) {
    ListCell *cell;
    StringInfoData targetColumn;
    StringInfoData trainOnColumns;
    char *tableName = NULL;
    char *whereClause = "<DEPRECATED>";

    initStringInfo(&targetColumn);
    initStringInfo(&trainOnColumns);

    /* Extract the column names from targetList and combine them into a single string */
    foreach(cell, stmt->targetList) {
        ResTarget *res = (ResTarget *) lfirst(cell);
        if (res == NULL || res->val == NULL) {
            elog(ERROR, "Null target column in statement");
            return InvalidObjectAddress;
        }
        char *colname = FigureColname(res->val);
        if (colname == NULL) {
            elog(ERROR, "Null column name in target list");
            return InvalidObjectAddress;
        }
        appendStringInfo(&targetColumn, "%s", colname);
    }
    targetColumn.data[targetColumn.len] = '\0';

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

    /* Extract the TrainOnSpec */
    if (stmt->trainOnSpec != NULL) {
        NeurDBTrainOnSpec *trainOnSpec = (NeurDBTrainOnSpec *) stmt->trainOnSpec;
        foreach(cell, trainOnSpec->trainOn) {
            Node *columnName = (Node *) lfirst(cell);
            if (columnName->type == T_A_Star) {
                elog(DEBUG1, "Train on all columns");
                break;
            }
            appendStringInfo(&trainOnColumns, "%s,", strVal(columnName));
        }
    } else {
        elog(DEBUG1, "No TrainOnSpec provided");
    }
    trainOnColumns.data[trainOnColumns.len - 1] = '\0';

    /* Execute the UDF with extracted columns, table name, and where clause */
    exec_udf(tableName, trainOnColumns.data, targetColumn.data, whereClause);

    return InvalidObjectAddress;
}
