#include "postgres.h"

#include "neurdb/guc.h"

#include "executor/executor.h"
#include "executor/spi.h"
#include "nodes/execnodes.h"

#include "access/relation.h"
#include "access/heapam.h"
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


static char *trainingFuncName = "nr_train";
#define TRAINING_PARAMS_ARRAY_SIZE 9
static Oid	trainingArgTypes[TRAINING_PARAMS_ARRAY_SIZE] = {TEXTOID, TEXTOID, INT4OID, INT4OID, INT4OID, INT4OID, TEXTARRAYOID, TEXTOID, INT4OID};

static char *inferenceFuncName = "nr_inference";
#define INFERENCE_PARAMS_ARRAY_SIZE 9
static Oid	inferenceArgTypes[INFERENCE_PARAMS_ARRAY_SIZE] = {TEXTOID, INT4OID, TEXTOID, INT4OID, INT4OID, INT4OID, TEXTARRAYOID, TEXTOID, INT4OID};

static char *initFuncName = "nr_pipeline_init";
#define INIT_PARAMS_ARRAY_SIZE 9
static Oid	initArgTypes[INIT_PARAMS_ARRAY_SIZE] =
{
	TEXTOID, //model name
	TEXTOID, //table name
	INT4OID, //batch size
	INT4OID, //epoch
	INT4OID, //nfeat
	TEXTARRAYOID, //feature names
	TEXTOID, //target
	INT4OID, //type
	ANYELEMENTOID // tupdesc
};

static char *pushSlotFuncName = "nr_pipeline_push_slot";
#define PUSHSLOT_PARAMS_ARRAY_SIZE 2
static Oid	pushSlotArgTypes[PUSHSLOT_PARAMS_ARRAY_SIZE] =
{
	ANYELEMENTOID, //slot
	BOOLOID // flush
};

static char *stateChangeFuncName = "nr_pipeline_state_change";
#define STATECHANGE_PARAMS_ARRAY_SIZE 1
static Oid	stateChangeArgTypes[STATECHANGE_PARAMS_ARRAY_SIZE] =
{
	BOOLOID // to inference
};

static char *closeFuncName = "nr_pipeline_close";
#define CLOSE_PARAMS_ARRAY_SIZE 0
static Oid	closeArgTypes[CLOSE_PARAMS_ARRAY_SIZE] = {};


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

static void
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

static void
insert_float8_to_tup_output(TupOutputState *tstate, float8 value, List *id_class_map, bool enable_debug)
{
	Datum		values[1];
	bool		nulls[1] = {0};

	values[0] = Float8GetDatum(value);
	do_tup_output(tstate, values, nulls);
}

static void
insert_cstring_to_tup_output(TupOutputState *tstate, float8 value, List *id_class_map, bool enable_debug)
{
	Datum		values[2];
	bool		nulls[2] = {0, 0};

	String	   *str_value = NULL;

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

#if 0
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
#endif

static char **
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


typedef struct
{
	Datum		value;
	bool		isnull;
}			UdfResult;

static UdfResult
call_udf_function(const char *funcName,
				  Oid *argTypes,
				  int nargs,
				  Datum *args,
				  bool *nulls)
{
	FmgrInfo	fmgrInfo;

	LOCAL_FCINFO(fcinfo, FUNC_MAX_ARGS);
	Oid			funcOid;
	UdfResult	result;

	funcOid = LookupFuncName(list_make1(makeString(funcName)), nargs, argTypes, false);
	if (!OidIsValid(funcOid))
		elog(ERROR, "Function %s not found", funcName);

	fmgr_info(funcOid, &fmgrInfo);
	InitFunctionCallInfoData(*fcinfo, &fmgrInfo, nargs, InvalidOid, NULL, NULL);

	for (int i = 0; i < nargs; i++)
	{
		fcinfo->args[i].value = args[i];
		fcinfo->args[i].isnull = nulls ? nulls[i] : false;
	}

	result.value = FunctionCallInvoke(fcinfo);
	result.isnull = fcinfo->isnull;

	if (result.isnull)
		elog(DEBUG2, "%s returned NULL", funcName);

	return result;
}

#if 0
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
static void
exec_udf(PredictType type,
		 const char *model,
		 const char *table,
		 const char *trainColumns,
		 const char *targetColumn,
		 const char *whereClause,
		 DestReceiver *dest)
{
	ArrayType  *trainColumnArray = get_train_columns_array(table, targetColumn, trainColumns);
	int			modelId = 0;

	/* model does not exist, training first */
	if (modelId == 0)
	{
		Datum		args[TRAINING_PARAMS_ARRAY_SIZE];
		bool		nulls[TRAINING_PARAMS_ARRAY_SIZE] = {false};

		args[0] = CStringGetTextDatum(model);
		args[1] = CStringGetTextDatum(table);
		args[2] = Int32GetDatum(NrTaskBatchSize);
		args[3] = Int32GetDatum(NrTaskNumBatches);
		args[4] = Int32GetDatum(NrTaskEpoch);
		args[5] = Int32GetDatum(NrTaskMaxFeatures);
		args[6] = PointerGetDatum(trainColumnArray);
		args[7] = CStringGetTextDatum(targetColumn);
		args[8] = Int32GetDatum(type);

		UdfResult	res = call_udf_function(trainingFuncName,
											trainingArgTypes,
											TRAINING_PARAMS_ARRAY_SIZE,
											args, nulls);

		if (!res.isnull)
		{
			int			result = DatumGetInt32(res.value);

			elog(NOTICE, "Training result: %d", result);
			modelId = result;
		}
		else
		{
			elog(NOTICE, "Training result is NULL");
		}
	}

	/* inference */
	ArrayType  *trainColumnArray = get_train_columns_array(table, targetColumn, trainColumns);
	Datum		args[INFERENCE_PARAMS_ARRAY_SIZE];
	bool		nulls[INFERENCE_PARAMS_ARRAY_SIZE] = {false};

	args[0] = CStringGetTextDatum(model);
	args[1] = Int32GetDatum(modelId);
	args[2] = CStringGetTextDatum(table);
	args[3] = Int32GetDatum(NrTaskBatchSize);
	args[4] = Int32GetDatum(NrTaskEpoch);
	args[5] = Int32GetDatum(NrTaskMaxFeatures);
	args[6] = PointerGetDatum(trainColumnArray);
	args[7] = CStringGetTextDatum(targetColumn);
	args[8] = Int32GetDatum(type);

	UdfResult	inferRes = call_udf_function(inferenceFuncName,
											 inferenceArgTypes,
											 INFERENCE_PARAMS_ARRAY_SIZE,
											 args, nulls);

	if (!inferRes.isnull)
	{
		NeurDBInferenceResult *result = (NeurDBInferenceResult *) DatumGetPointer(inferRes.value);

		return_table(dest, result);
	}
	else
	{
		elog(DEBUG2, "Inference result is NULL");
	}
}
#endif


/** 
 * Fill the given slot with one tuple derived from a NeurDBInferenceResult.
 * The slot already has a valid TupleDesc matching the node’s output schema.
 * Returns the same slot pointer.
 */
static TupleTableSlot *
build_result_slot(const NeurDBInferenceResult *result, TupleTableSlot *slot)
{
    bool  is_float = (result->typeoid == FLOAT8OID);
    Datum values[2];
    bool  nulls[2] = {false, false};
    double parsed = 0.0;
    char *endptr = NULL;
    const char *s = result->result;

    // ExecClearTuple(slot);

    /* parse first number from result->result */
    if (s && *s)
	{
        parsed = atof(s);
	}
	else
	{
		elog(ERROR, "Failed to parse float");
	}

    if (is_float)
    {
        if (endptr == s)
            nulls[0] = true;
        else
            values[0] = Float8GetDatum(parsed);

        // ExecStoreVirtualTuple(slot);
        slot->tts_values[0] = values[0];
        slot->tts_isnull[0] = nulls[0];
		slot->tts_tupleDescriptor->attrs[0].atttypid = FLOAT8OID;
    }
    else if (result->typeoid == TEXTOID)
    {
        /* derive label and debug value */
        const char *label = NULL;

        if (endptr != s && result->id_class_map && list_length(result->id_class_map) >= 2)
        {
            label = (parsed > 0)
                ? strVal(list_nth(result->id_class_map, 1))
                : strVal(list_nth(result->id_class_map, 0));
        }
        else
        {
            label = result->result ? result->result : "";
        }

        values[0] = CStringGetTextDatum(label);
        values[1] = Float8GetDatum(parsed);
        nulls[0] = false;
        nulls[1] = (endptr == s);

        // ExecStoreVirtualTuple(slot);
        slot->tts_values[0] = values[0];
        slot->tts_values[1] = values[1];
        slot->tts_isnull[0] = nulls[0];
        slot->tts_isnull[1] = nulls[1];
    }
    else
    {
        elog(ERROR, "Unsupported typeoid in build_result_slot: %u", result->typeoid);
    }

    return slot;
}


static TupleTableSlot *
ExecNeurDBPredict(PlanState *pstate)
{
	NeurDBPredictState *predictstate = (NeurDBPredictState *) pstate;
	PlanState  *outerPlan;
	TupleTableSlot *slot;

	outerPlan = outerPlanState(predictstate);

	for (;;)
	{
		/* TEMP: Return dummy result */
		slot = ExecProcNode(outerPlan);
		if (TupIsNull(slot))
		{
			if (predictstate->nrpstate == NEURDBPREDICT_TRAIN)
			{
				predictstate->nrpstate = NEURDBPREDICT_INFERENCE;

				/* tell nr_pipeline to change state */
				elog(DEBUG1, "change state to inference");
				
				Oid			funcOid = LookupFuncName(
					list_make1(makeString(stateChangeFuncName)), 
					STATECHANGE_PARAMS_ARRAY_SIZE, 
					stateChangeArgTypes,
					false
				);
				if (!OidIsValid(funcOid))
					elog(ERROR, "Function %s not found", stateChangeFuncName);

				OidFunctionCall1(funcOid, BoolGetDatum(true));

				ExecReScan(outerPlan);
				continue;
			}
			else
			{
				/* end inference */
				return NULL;
			}
		}

		Datum		args[INFERENCE_PARAMS_ARRAY_SIZE];
		bool		nulls[INFERENCE_PARAMS_ARRAY_SIZE] = {false};

		args[0] = PointerGetDatum(slot);

		UdfResult	pushSlotRes = call_udf_function(pushSlotFuncName,
													pushSlotArgTypes,
													PUSHSLOT_PARAMS_ARRAY_SIZE,
													args, nulls);
		
		if (predictstate->nrpstate == NEURDBPREDICT_INFERENCE)
		{
			NeurDBInferenceResult *result = (NeurDBInferenceResult *) DatumGetPointer(pushSlotRes.value);

			/* apply projection early to manipulate the tupledesc */
			predictstate->ps.ps_ExprContext->ecxt_outertuple = slot;
			slot = ExecProject(predictstate->ps.ps_ProjInfo);

			build_result_slot(result, slot);
			break;
		}

	}

	return slot;
}

static ArrayType *
get_train_columns_array(const char *table, const char *targetColumn, const char *trainColumns)
{
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

	return construct_array(trainColumnDatums, nTrainColumns, TEXTOID, -1, false, 'i');
}


static StringInfoData
construct_target_columns(List *targetList)
{
	StringInfoData result;
	ListCell   *cell;

	initStringInfo(&result);

	foreach(cell, targetList)
	{
		ResTarget  *res = (ResTarget *) lfirst(cell);

		if (res == NULL || res->val == NULL)
		{
			elog(ERROR, "Null target column in statement");
		}

		char	   *colname = FigureColname(res->val);

		if (colname == NULL)
		{
			elog(ERROR, "Null column name in target list");
		}

		appendStringInfo(&result, "%s", colname);
	}

	result.data[result.len] = '\0';

	return result;
}

static char *
_temp_extract_model_name(NeurDBTrainOnSpec *trainOnSpec)
{
	if (trainOnSpec == NULL)
	{
		elog(DEBUG1, "No TrainOnSpec provided");
		return NrModelName;
	}

	if (strlen(trainOnSpec->modelName) == 0)
	{
		elog(WARNING, "No model name provided. Use config NrModelName: %s", NrModelName);
		return NrModelName;

	}

	elog(WARNING, "User specified model name: %s", trainOnSpec->modelName);
	return trainOnSpec->modelName;
}

static StringInfoData 
_temp_extract_train_on_columns(List *trainOn)
{
	StringInfoData result;
	initStringInfo(&result);

	ListCell   *cell;
	foreach(cell, trainOn)
	{
		TargetEntry	   *column = (ResTarget *) lfirst(cell);
		appendStringInfo(&result, "%s,", column->resname);
	}

	if (result.len > 0)
	{
		result.data[result.len - 1] = '\0';
	}

	return result;
}


static char *
_temp_extract_table_name(List *fromClause)
{
	/* Extract the table name from fromClause */
	if (fromClause == NIL)
	{
		elog(ERROR, "No from clause in statement");
		return NULL;
	}

	RangeSubselect *rss = (RangeSubselect *) linitial(fromClause);

	if (rss == NULL)
	{
		elog(ERROR, "Null range variable in from clause");
		return NULL;
	}
	SelectStmt *selectStmt = (SelectStmt *) rss->subquery;

	if (selectStmt == NULL || selectStmt->fromClause == NIL)
	{
		elog(ERROR, "No from clause in statement");
		return NULL;
	}

	RangeVar   *rv = (RangeVar *) linitial(selectStmt->fromClause);
	char	   *table = rv->relname;

	elog(DEBUG1, "Extracted table name: %s", table);

	return table;
}


NeurDBPredictState *
ExecInitNeurDBPredict(NeurDBPredict * node, EState *estate, int eflags)
{
	NeurDBPredictState *predictstate;
	Plan	   *outerPlan;

	predictstate = makeNode(NeurDBPredictState);
	predictstate->ps.plan = (Plan *) node;
	predictstate->ps.plan->targetlist = node->predictTargetList;
	predictstate->ps.state = estate;
	predictstate->ps.ExecProcNode = ExecNeurDBPredict;

	/* predictstate->targetList = node->targetList; */
	/* predictstate->fromClause = node->fromClause; */
	predictstate->stmt = node->stmt;

	/*
	 * To use projection, need ExprContext
	 */
	ExecAssignExprContext(estate, &predictstate->ps);

	/*
	 * initialize outer plan
	 */
	outerPlan = outerPlan(node);
	outerPlanState(predictstate) = ExecInitNode(outerPlan, estate, eflags);

	/*
	 * Initialize result tuple slot
	 */
	ExecInitResultTupleSlotTL(&predictstate->ps, &TTSOpsVirtual);

	/*
	 * initialize projection info
	 */
	predictstate->ps.ps_ProjInfo =
		ExecBuildProjectionInfo(node->predictTargetList,
								predictstate->ps.ps_ExprContext,
								predictstate->ps.ps_ResultTupleSlot,
								predictstate,
								ExecTypeFromTL(node->predictTargetList));

	StringInfoData targetColumn = construct_target_columns(node->predictTargetList);

	Datum		args[INFERENCE_PARAMS_ARRAY_SIZE];
	bool		nulls[INFERENCE_PARAMS_ARRAY_SIZE] = {false};

	char	   *table = _temp_extract_table_name(predictstate->stmt->fromClause);
	char	   *model = _temp_extract_model_name(predictstate->stmt->trainOnSpec);
	StringInfoData trainOnColumns = _temp_extract_train_on_columns(node->trainOn);

	ArrayType  *trainColumnArray = get_train_columns_array(table, targetColumn.data, trainOnColumns.data);

	args[0] = CStringGetTextDatum(model);
	args[1] = CStringGetTextDatum(table);
	args[2] = Int32GetDatum(NrTaskBatchSize);
	args[3] = Int32GetDatum(NrTaskNumBatches);
	args[4] = Int32GetDatum(NrTaskMaxFeatures);
	args[5] = PointerGetDatum(trainColumnArray);
	args[6] = CStringGetTextDatum(targetColumn.data);
	args[7] = Int32GetDatum(predictstate->stmt->kind);
	args[8] = PointerGetDatum(ExecTypeFromTL(node->trainOn));

	UdfResult	initRes = call_udf_function(initFuncName,
											initArgTypes,
											INIT_PARAMS_ARRAY_SIZE,
											args, nulls);

	if (!initRes.isnull)
	{
		bool		is_inference = DatumGetBool(initRes.value);

		if (is_inference)
		{
			predictstate->nrpstate = NEURDBPREDICT_INFERENCE;
		}
		else
		{
			predictstate->nrpstate = NEURDBPREDICT_TRAIN;
		}
	}

	return predictstate;
}

/* ----------------------------------------------------------------
 *		ExecEndNeurDBPredict
 *
 *		This shuts down the subplan and frees resources allocated
 *		to this node.
 * ----------------------------------------------------------------
 */
void
ExecEndNeurDBPredict(NeurDBPredictState * node)
{
	ExecFreeExprContext(&node->ps);
	ExecEndNode(outerPlanState(node));

	Oid			funcOid = LookupFuncName(list_make1(makeString(closeFuncName)), 0, NULL, false);
	if (!OidIsValid(funcOid))
		elog(ERROR, "Function %s not found", closeFuncName);

	/* ensure SPI available */
	// if (SPI_connect() != SPI_OK_CONNECT)
	// 	elog(ERROR, "SPI_connect failed");

	OidFunctionCall0(funcOid);
	
	// if (SPI_finish() != SPI_OK_FINISH)
	// 	elog(ERROR, "SPI_finish failed");

	elog(DEBUG1, "NeurDB prediction end");
}

void
ExecReScanNeurDBPredict(NeurDBPredictState * node)
{
	PlanState  *outerPlan = outerPlanState(node);

	/*
	 * If chgParam of subnode is not null then plan will be re-scanned by
	 * first ExecProcNode.
	 */
	if (outerPlan && outerPlan->chgParam == NULL)
		ExecReScan(outerPlan);
}
