#include "postgres.h"

#include "predict.h"

#include "access/relation.h"
#include "access/heapam.h"
#include "executor/executor.h"
#include "parser/parse_func.h"
#include "parser/parse_node.h"
#include "parser/parse_target.h"
#include "catalog/pg_type.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "neurdb/predict.h"
#include "nodes/primnodes.h"
#include "nodes/makefuncs.h"

/**
 * Static (fixed) look-up variables
 */
char	   *modelLookupFuncName = "nr_model_lookup";
#define MODEL_LOOKUP_PARAMS_ARRAY_SIZE 3
Oid			modelLookupArgTypes[MODEL_LOOKUP_PARAMS_ARRAY_SIZE] = {TEXTOID, TEXTARRAYOID, TEXTOID};

#define TRAINING_PARAMS_ARRAY_SIZE 9
char	   *trainingFuncName = "nr_train";
Oid			trainingArgTypes[TRAINING_PARAMS_ARRAY_SIZE] = {TEXTOID, TEXTOID, INT4OID, INT4OID, INT4OID, INT4OID, TEXTARRAYOID, TEXTOID, INT4OID};

#define INFERENCE_PARAMS_ARRAY_SIZE 8
char	   *inferenceFuncName = "nr_inference";
Oid			inferenceArgTypes[INFERENCE_PARAMS_ARRAY_SIZE] = {TEXTOID, INT4OID, TEXTOID, INT4OID, INT4OID, INT4OID, TEXTARRAYOID, INT4OID};


static List *
split_columns(const char *columns)
{
	List	   *result = NIL;
	char	   *token = strtok(columns, ",");

	while (token != NULL)
	{
		result = lappend(result, makeString(token));
		token = strtok(NULL, ",");
	}
	return result;
}

static void
set_false_to_all_params(NullableDatum *args, int size)
{
	for (int i = 0; i < size; i++)
	{
		args[i].isnull = false;
	}
}

void
parseDoubles(const NeurDBInferenceResult * result,
			 void (*callback) (TupOutputState *, double, List *, bool),
			 TupOutputState *tstate,
			 bool enable_debug)
{
	char		buffer[64];

	/* Buffer to accumulate characters */
	int			bufIndex = 0;

	const char *str = result->result;

	while (*str)
	{
		if (*str == ' ')
		{
			/* When space is found, terminate the buffer and parse */
			if (bufIndex > 0)
			{
				/* Null terminate the string */
				buffer[bufIndex] = '\0';

				double		value = atof(buffer);

				/* Convert to double */
				/* printf("Found double: %f\n", value); */
				callback(tstate, value, result->id_class_map, enable_debug);

				/* Reset buffer index */
				bufIndex = 0;
			}
		}
		else
		{
			/* Add character to buffer if there's space */
			if (bufIndex < sizeof(buffer) - 1)
			{
				buffer[bufIndex++] = *str;
			}
		}
		/* Move to next character */
		str++;
	}

	/* Handle the last number if string doesn't end with space */
	if (bufIndex > 0)
	{
		buffer[bufIndex] = '\0';
		double		value = atof(buffer);

		/* printf("Found double: %f\n", value); */
		callback(tstate, value, result->id_class_map, enable_debug);
	}
}

void
insert_float8_to_tup_output(TupOutputState *tstate, float8 value, List *id_class_map, bool enable_debug)
{
	Datum		values[1];
	bool		nulls[1] = {0};

	values[0] = Float8GetDatum(value);
	do_tup_output(tstate, values, nulls);
}

void
insert_cstring_to_tup_output(TupOutputState *tstate, float8 value, List *id_class_map, bool enable_debug)
{
	Datum		values[2];
	bool		nulls[2] = {0, 0};

	String *str_value = NULL;

	/* TODO: support multiclass classification */
	if (value > 0)
	{
		values[0] = CStringGetTextDatum(strVal(list_nth(id_class_map, 1)));
	}
	else
	{
		values[0] = CStringGetTextDatum(strVal(list_nth(id_class_map, 0)));
	}

	if (enable_debug)
	{
		values[1] = Float8GetDatum(value);
	}

	do_tup_output(tstate, values, nulls);
}


static void
return_table(DestReceiver *dest, const NeurDBInferenceResult * result)
{
	TupOutputState *tstate;
	TupleDesc	tupdesc;
	ListCell   *lc;

	if (result->typeoid == FLOAT8OID)
	{
		tupdesc = CreateTemplateTupleDesc(1);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "result", result->typeoid, -1, 0);

		tstate = begin_tup_output_tupdesc(dest, tupdesc, &TTSOpsVirtual);

		parseDoubles(result, &insert_float8_to_tup_output, tstate, false);
	}
	else if (result->typeoid == TEXTOID)
	{
		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "result", result->typeoid, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "_dbg_value", FLOAT8OID, -1, 0);


		tstate = begin_tup_output_tupdesc(dest, tupdesc, &TTSOpsVirtual);

		parseDoubles(result, &insert_cstring_to_tup_output, tstate, true);
	}
	else
	{
		elog(ERROR, "Unsupported data type");
	}

	end_tup_output(tstate);

}

char	  **
get_column_names(const char *schema_name, const char *table_name, const char *exclude, int *num_included_out)
{
	int			max_num_columns = 100;

	char	  **column_names = (char **) palloc(sizeof(char *) * max_num_columns);

	/* Get table Oid */
	RangeVar   *rangeVar = makeRangeVar((char *) schema_name, (char *) table_name, -1);
	Oid			tableOid = RangeVarGetRelid(rangeVar, AccessShareLock, false);

	if (tableOid == InvalidOid)
	{
		elog(ERROR, "Table %s.%s not found", schema_name, table_name);
	}

	/* Open relation and get column names */
	Relation	rel = relation_open(tableOid, AccessShareLock);
	TupleDesc	tupleDesc = rel->rd_att;

	if (tupleDesc->natts > max_num_columns)
	{
		elog(ERROR, "Too many columns in table %s.%s", schema_name, table_name);
	}

	int			num_included = 0;

	for (int i = 0; i < tupleDesc->natts; i++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupleDesc, i);

		if (!attr->attisdropped)
		{
			char	   *colName = NameStr(attr->attname);

			elog(DEBUG1, "Column %d: %s", i + 1, colName);

			if (strcmp(colName, exclude) != 0)
			{
				column_names[num_included] = strcpy((char *) palloc(strlen(colName) + 1), colName);
				num_included++;
			}
		}
	}

	*num_included_out = num_included;

	relation_close(rel, AccessShareLock);

	return column_names;
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
exec_udf(PredictType type,
		 const char *model,
		 const char *table,
		 const char *trainColumns,
		 const char *targetColumn,
		 const char *whereClause,
		 DestReceiver *dest)
{
	/* lookup function call infos */
	FmgrInfo	modelLookupFmgrInfo;

	/* fcinfo for lookup function */
	LOCAL_FCINFO(modelLookupFCInfo, FUNC_MAX_ARGS);
	Datum		modelLookupResult;

	Oid			modelLookupFuncOid = LookupFuncName(list_make1(makeString(modelLookupFuncName)), 3, modelLookupArgTypes, false);

	if (!OidIsValid(modelLookupFuncOid))
	{
		elog(ERROR, "Function %s not found", modelLookupFuncName);
		return;
	}

	fmgr_info(modelLookupFuncOid, &modelLookupFmgrInfo);
	InitFunctionCallInfoData(*modelLookupFCInfo, &modelLookupFmgrInfo, MODEL_LOOKUP_PARAMS_ARRAY_SIZE, InvalidOid, NULL, NULL);

	/* split columns into an array of text */
	List	   *trainColumnsList = split_columns(trainColumns);
	Datum	   *trainColumnDatums;
	int			nTrainColumns = 0;

	if (strlen(trainColumns) == 0)
	{
		/* all columns are used for training */

		/* get all column names */

		char	  **allColumns = get_column_names("public", table, targetColumn, &nTrainColumns);

		trainColumnDatums = (Datum *) palloc(sizeof(Datum) * nTrainColumns);
		for (int i = 0; i < nTrainColumns; i++)
		{
			trainColumnDatums[i] = CStringGetTextDatum(allColumns[i]);
		}

		for (int i = 0; i < nTrainColumns; i++)
		{
			pfree(allColumns[i]);
		}
	}
	else
	{
		nTrainColumns = list_length(trainColumnsList);
		trainColumnDatums = (Datum *) palloc(sizeof(Datum) * nTrainColumns);
		for (int i = 0; i < nTrainColumns; i++)
		{
			trainColumnDatums[i] = CStringGetTextDatum(strVal(list_nth(trainColumnsList, i)));
		}

	}
	ArrayType  *trainColumnArray = construct_array(trainColumnDatums, nTrainColumns, TEXTOID, -1, false, 'i');

	modelLookupFCInfo->args[0].value = CStringGetTextDatum(table);
	modelLookupFCInfo->args[1].value = PointerGetDatum(trainColumnArray);
	modelLookupFCInfo->args[2].value = CStringGetTextDatum(targetColumn);

	modelLookupResult = FunctionCallInvoke(modelLookupFCInfo);
	int			modelId = 0;

	/* modelLookupResult is a boolean */
	if (!modelLookupFCInfo->isnull)
	{
		modelId = DatumGetInt32(modelLookupResult);
	}

	/* model does not exist, training first */
	if (modelId == 0)
	{
		FmgrInfo	trainingFmgrInfo;

		/* fcinfo for training function */
		LOCAL_FCINFO(trainingFCInfo, FUNC_MAX_ARGS);
		Datum		trainingResult;

		Oid			trainingFuncOid = LookupFuncName(list_make1(makeString(trainingFuncName)),
													 TRAINING_PARAMS_ARRAY_SIZE, trainingArgTypes, false);

		if (!OidIsValid(trainingFuncOid))
		{
			elog(ERROR, "Function %s not found", trainingFuncName);
			return;
		}

		/* rebuild trainColumnArray since it's freed in nr_train() */
		ArrayType  *trainColumnArray = construct_array(trainColumnDatums, nTrainColumns, TEXTOID, -1, false, 'i');

		fmgr_info(trainingFuncOid, &trainingFmgrInfo);
		InitFunctionCallInfoData(*trainingFCInfo, &trainingFmgrInfo, TRAINING_PARAMS_ARRAY_SIZE, InvalidOid, NULL, NULL);

		trainingFCInfo->args[0].value = CStringGetTextDatum(model);
		trainingFCInfo->args[1].value = CStringGetTextDatum(table);
		trainingFCInfo->args[2].value = Int32GetDatum(NrTaskBatchSize);
		trainingFCInfo->args[3].value = Int32GetDatum(NrTaskNumBatches);
		trainingFCInfo->args[4].value = Int32GetDatum(NrTaskEpoch);
		trainingFCInfo->args[5].value = Int32GetDatum(NrTaskMaxFeatures);
		trainingFCInfo->args[6].value = PointerGetDatum(trainColumnArray);
		trainingFCInfo->args[7].value = CStringGetTextDatum(targetColumn);
		trainingFCInfo->args[8].value = Int32GetDatum(type);

		set_false_to_all_params(trainingFCInfo->args, TRAINING_PARAMS_ARRAY_SIZE);

		trainingResult = FunctionCallInvoke(trainingFCInfo);
		if (!trainingFCInfo->isnull)
		{
			int			result = DatumGetInt32(trainingResult);

			elog(NOTICE, "Training result: %d", result);
			modelId = result;
		}
		else
		{
			elog(NOTICE, "Training result is NULL");
		}
	}

	/* inference */
	FmgrInfo	inferenceFmgrInfo;

	/* fcinfo for inference function */
	LOCAL_FCINFO(inferenceFCInfo, FUNC_MAX_ARGS);
	Datum		inferenceResult;

	Oid			inferenceFuncOid = LookupFuncName(list_make1(makeString(inferenceFuncName)),
												  INFERENCE_PARAMS_ARRAY_SIZE, inferenceArgTypes, false);

	if (!OidIsValid(inferenceFuncOid))
	{
		elog(ERROR, "Function %s not found", inferenceFuncName);
		return;
	}

	fmgr_info(inferenceFuncOid, &inferenceFmgrInfo);
	InitFunctionCallInfoData(*inferenceFCInfo, &inferenceFmgrInfo, INFERENCE_PARAMS_ARRAY_SIZE, InvalidOid, NULL, NULL);

	inferenceFCInfo->args[0].value = CStringGetTextDatum(model);
	inferenceFCInfo->args[1].value = Int32GetDatum(modelId);
	inferenceFCInfo->args[2].value = CStringGetTextDatum(table);
	inferenceFCInfo->args[3].value = Int32GetDatum(NrTaskBatchSize);
	inferenceFCInfo->args[4].value = Int32GetDatum(NrTaskNumBatches);
	inferenceFCInfo->args[5].value = Int32GetDatum(NrTaskMaxFeatures);
	inferenceFCInfo->args[6].value = PointerGetDatum(trainColumnArray);
	inferenceFCInfo->args[7].value = Int32GetDatum(type);

	set_false_to_all_params(inferenceFCInfo->args, INFERENCE_PARAMS_ARRAY_SIZE);

	inferenceResult = FunctionCallInvoke(inferenceFCInfo);
	if (!inferenceFCInfo->isnull)
	{
		NeurDBInferenceResult *result = (NeurDBInferenceResult *) DatumGetPointer(inferenceResult);

		return_table(dest, result);
	}
	else
	{
		elog(DEBUG2, "Inference result is NULL");
	}
}

/*
* ExecPredictStmt --- Execution for node NeurDBPredictStmt defined in include/nodes/parsenodes.h
*
* NeurDBPredictStmt: Node structure in include/nodes/parsenodes.h
*/
ObjectAddress
ExecPredictStmt(NeurDBPredictStmt * stmt, ParseState *pstate, const char *whereClauseString, DestReceiver *dest)
{
	elog(DEBUG1, "[ExecPredictStmt] In the ExecPredictStmt");

	ListCell   *cell;
	StringInfoData targetColumn;
	StringInfoData trainOnColumns;
	char	   *modelName = NULL;
	char	   *tableName = NULL;
	char	   *whereClause = "<DEPRECATED>";

	initStringInfo(&targetColumn);
	initStringInfo(&trainOnColumns);

	/* if (stmt->kind == PREDICT_CLASS) */
	/* { */
	/* elog(ERROR, "PREDICT CLASS OF is not implemented"); */
	/* return InvalidObjectAddress; */
	/* } */

	/*
	 * Extract the column names from targetList and combine them into a single
	 * string
	 */
	foreach(cell, stmt->targetList)
	{
		ResTarget  *res = (ResTarget *) lfirst(cell);

		if (res == NULL || res->val == NULL)
		{
			elog(ERROR, "Null target column in statement");
			return InvalidObjectAddress;
		}
		char	   *colname = FigureColname(res->val);

		if (colname == NULL)
		{
			elog(ERROR, "Null column name in target list");
			return InvalidObjectAddress;
		}
		appendStringInfo(&targetColumn, "%s", colname);
	}
	targetColumn.data[targetColumn.len] = '\0';

	/* Extract the table name from fromClause */
	if (stmt->fromClause != NIL)
	{
		RangeVar   *rv = (RangeVar *) linitial(stmt->fromClause);

		if (rv == NULL)
		{
			elog(ERROR, "Null range variable in from clause");
			return InvalidObjectAddress;
		}
		tableName = rv->relname;
		elog(DEBUG1, "Extracted table name: %s", tableName);
	}
	else
	{
		elog(ERROR, "No from clause in statement");
		return InvalidObjectAddress;
	}

	/* Extract the TrainOnSpec */
	if (stmt->trainOnSpec != NULL)
	{
		NeurDBTrainOnSpec *trainOnSpec = (NeurDBTrainOnSpec *) stmt->trainOnSpec;

		foreach(cell, trainOnSpec->trainOn)
		{
			Node	   *columnName = (Node *) lfirst(cell);

			if (columnName->type == T_A_Star)
			{
				elog(DEBUG1, "Train on all columns");
				resetStringInfo(&trainOnColumns);
				break;
			}
			appendStringInfo(&trainOnColumns, "%s,", strVal(columnName));
		}

		if (strlen(trainOnSpec->modelName) > 0)
		{
			elog(WARNING, "User specified model name: %s", trainOnSpec->modelName);
			modelName = trainOnSpec->modelName;
		}
		else
		{
			elog(WARNING, "No model name provided. Use config NrModelName: %s", NrModelName);
			modelName = NrModelName;
		}
	}
	else
	{
		elog(DEBUG1, "No TrainOnSpec provided");
	}

	if (trainOnColumns.len > 0)
	{
		trainOnColumns.data[trainOnColumns.len - 1] = '\0';
	}

	/* Execute the UDF with extracted columns, table name, and where clause */
	exec_udf(stmt->kind, modelName, tableName, trainOnColumns.data, targetColumn.data, whereClause, dest);

	return InvalidObjectAddress;
}
