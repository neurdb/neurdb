#include "postgres.h"

#include "neurdb/guc.h"

#include "executor/executor.h"
#include "executor/spi.h"
#include "nodes/execnodes.h"

#include "access/relation.h"
#include "access/heapam.h"
#include "funcapi.h"
#include "nodes/nodeFuncs.h"
#include "parser/parse_func.h"
#include "parser/parse_node.h"
#include "parser/parse_target.h"
#include "catalog/pg_type.h"
#include "fmgr.h"
#include "lib/ilist.h"
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
#define INIT_PARAMS_ARRAY_SIZE 10
static Oid	initArgTypes[INIT_PARAMS_ARRAY_SIZE] =
{
	TEXTOID, //model name
	TEXTOID, //table name
	INT4OID, //batch size
	INT4OID, //epoch (number of training epochs)
	INT4OID, //n_batches
	INT4OID, //nfeat
	TEXTARRAYOID, //feature names
	TEXTOID, //target
	INT4OID, //type
	ANYELEMENTOID // tupdesc
};

static char *pushSlotFuncName = "nr_pipeline_push_slot";
#define PUSHSLOT_PARAMS_ARRAY_SIZE 3
static Oid	pushSlotArgTypes[PUSHSLOT_PARAMS_ARRAY_SIZE] =
{
	ANYELEMENTOID, //slot
	INT4OID, //num_slot
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

typedef struct result_node
{
	dlist_node	node;
	double		value;
}			result_node;

static void
insert_result_to_cache(NeurDBPredictState * pstate, double value)
{
	result_node *node = palloc(sizeof(result_node));

	node->value = value;
	dclist_push_tail(&pstate->result_cache, &node->node);
}


static void
parse_result_to_cache(const NeurDBInferenceResult * result,
					  NeurDBPredictState * pstate,
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
				insert_result_to_cache(pstate, value);

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
		insert_result_to_cache(pstate, value);
	}
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

		parse_result_to_cache(result, &insert_float8_to_tup_output, tstate, false);
	}
	else if (result->typeoid == TEXTOID)
	{
		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "result", result->typeoid, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "_dbg_value", FLOAT8OID, -1, 0);


		tstate = begin_tup_output_tupdesc(dest, tupdesc, &TTSOpsVirtual);

		parse_result_to_cache(result, &insert_cstring_to_tup_output, tstate, true);
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
build_result_slot(double value, bool is_float, List *id_class_map, TupleTableSlot *slot)
{
	ExecClearTuple(slot);

	if (is_float)
	{
		slot->tts_values[0] = Float8GetDatum(value);
		slot->tts_isnull[0] = false;
		slot->tts_tupleDescriptor->attrs[0].atttypid = FLOAT8OID;
	}
	else
	{
		/* derive label and debug value */
		const char *label = NULL;

		if (id_class_map && list_length(id_class_map) >= 2)
		{
			label = (value > 0)
				? strVal(list_nth(id_class_map, 1))
				: strVal(list_nth(id_class_map, 0));
		}
		else
		{
			label = "";
		}

		slot->tts_values[0] = CStringGetTextDatum(label);
		slot->tts_values[1] = Float8GetDatum(value);
		slot->tts_isnull[0] = false;
		slot->tts_isnull[1] = false;
		slot->tts_tupleDescriptor->attrs[0].atttypid = TEXTOID;
	}

	ExecStoreVirtualTuple(slot);

	return slot;
}

static void
reset_slot_cache(NeurDBPredictState * predictstate)
{
	for (int i = 0; i < predictstate->slot_cache_size; i++)
	{
		ExecClearTuple(predictstate->slot_cache[i]);
		pfree(predictstate->slot_cache[i]);
	}
	predictstate->slot_cache_size = 0;
}

static void
add_slot_to_cache(NeurDBPredictState * predictstate, TupleTableSlot *slot)
{
	TupleTableSlot *slot_copy = MakeTupleTableSlot(slot->tts_tupleDescriptor, &TTSOpsVirtual);

	ExecCopySlot(slot_copy, slot);
	predictstate->slot_cache[predictstate->slot_cache_size++] = slot_copy;
	ReleaseTupleDesc(slot->tts_tupleDescriptor);
}

static void
_call_pipeline_close()
{
	/* tell nr_pipeline to close the connection */
	elog(DEBUG1, "close connection");

	Oid			funcOid = LookupFuncName(list_make1(makeString(closeFuncName)),
										 CLOSE_PARAMS_ARRAY_SIZE,
										 closeArgTypes,
										 false);

	if (!OidIsValid(funcOid))
		elog(ERROR, "Function %s not found", stateChangeFuncName);

	OidFunctionCall0(funcOid);
}


static TupleTableSlot *
ExecNeurDBPredict(PlanState *pstate)
{
	NeurDBPredictState *predictstate = (NeurDBPredictState *) pstate;
	PlanState  *outerPlan;
	TupleTableSlot *slot;

	outerPlan = outerPlanState(predictstate);

	predictstate->is_final = false;

	for (;;)
	{
		switch (predictstate->nrpstate)
		{
			case NEURDBPREDICT_TRAIN_COLLECT:
				{
					/* if slot is full, send it to nr_pipeline */
					if (predictstate->slot_cache_size >= NrTaskBatchSize)
					{
						predictstate->nrpstate = NEURDBPREDICT_TRAIN_SEND;
						continue;
					}

					/* execute the outer plan to get new input */
					slot = ExecProcNode(outerPlan);
					if (TupIsNull(slot))
					{
						predictstate->is_final = true;
						predictstate->nrpstate = NEURDBPREDICT_TRAIN_SEND;
						continue;
					}

					/* cache not full, add slot to slot_cache */
					add_slot_to_cache(predictstate, slot);
				}
				break;

			case NEURDBPREDICT_TRAIN_SEND:
				{
					/*
					 * if slot_cache is empty, it means that the number of
					 * tuples is divisible by NrTaskBatchSize, and
					 * TupIsNull(slot) is true when the cache is empty.
					 */
					if (predictstate->slot_cache_size <= 0)
					{
						predictstate->nrpstate = NEURDBPREDICT_TRAIN_END;
						continue;
					}

					/* if slot_cache is not empty, send it to AI engine */
					Datum		args[PUSHSLOT_PARAMS_ARRAY_SIZE];
					bool		nulls[PUSHSLOT_PARAMS_ARRAY_SIZE] = {false};

					args[0] = PointerGetDatum(predictstate->slot_cache);
					args[1] = Int32GetDatum(predictstate->slot_cache_size);
					args[2] = BoolGetDatum(true);

					UdfResult	pushSlotRes = call_udf_function(pushSlotFuncName,
																pushSlotArgTypes,
																PUSHSLOT_PARAMS_ARRAY_SIZE,
																args, nulls);

					/* reset slot cache */
					reset_slot_cache(predictstate);
					predictstate->num_consumed = 0;

					if (predictstate->is_final)
					{
						/* the current batch is the last one */
						predictstate->nrpstate = NEURDBPREDICT_TRAIN_END;
					}
					else
					{
						/* go back to retrieve the next batch */
						predictstate->nrpstate = NEURDBPREDICT_TRAIN_COLLECT;
					}
				}
				break;

			case NEURDBPREDICT_TRAIN_END:
				{
					predictstate->curr_epoch += 1;
					elog(DEBUG1, "[NeurDBPredictState] Epoch: %d", predictstate->curr_epoch);

					/* rescan from the beginning */
					ExecReScan(outerPlan);

					if (predictstate->curr_epoch < NrTaskEpoch)
					{
						/* go back to collect */
						predictstate->is_final = false;
						predictstate->nrpstate = NEURDBPREDICT_TRAIN_COLLECT;
					}
					else
					{
						/* all epochs are done, go to inference */

						/* tell nr_pipeline to change state */
						elog(DEBUG1, "change state to inference");

						Oid			funcOid = LookupFuncName(list_make1(makeString(stateChangeFuncName)),
															STATECHANGE_PARAMS_ARRAY_SIZE,
															stateChangeArgTypes,
															false);

						if (!OidIsValid(funcOid))
							elog(ERROR, "Function %s not found", stateChangeFuncName);

						OidFunctionCall1(funcOid, BoolGetDatum(true));

						/* go to inference */
						predictstate->nrpstate = NEURDBPREDICT_INFERENCE_COLLECT;
					}
				}
				break;

			case NEURDBPREDICT_INFERENCE_COLLECT:
				{
					/* if slot is full, send it to nr_pipeline */
					if (predictstate->slot_cache_size >= NrTaskBatchSize)
					{
						predictstate->nrpstate = NEURDBPREDICT_INFERENCE_SEND;
						continue;
					}

					/* execute the outer plan to get new input */
					slot = ExecProcNode(outerPlan);
					if (TupIsNull(slot))
					{
						predictstate->is_final = true;
						predictstate->nrpstate = NEURDBPREDICT_INFERENCE_SEND;
						continue;
					}

					/* cache not full, add slot to slot_cache */
					add_slot_to_cache(predictstate, slot);
				}
				break;

			case NEURDBPREDICT_INFERENCE_SEND:
				{
					/*
					 * if slot_cache is empty, it means that the number of
					 * tuples is divisible by NrTaskBatchSize, and
					 * TupIsNull(slot) is true when the cache is empty.
					 */
					if (predictstate->slot_cache_size <= 0)
					{
						predictstate->nrpstate = NEURDBPREDICT_INFERENCE_RETURN;
						continue;
					}

					Datum		args[PUSHSLOT_PARAMS_ARRAY_SIZE];
					bool		nulls[PUSHSLOT_PARAMS_ARRAY_SIZE] = {false};

					args[0] = PointerGetDatum(predictstate->slot_cache);
					args[1] = Int32GetDatum(predictstate->slot_cache_size);
					args[2] = BoolGetDatum(true);

					UdfResult	pushSlotRes = call_udf_function(pushSlotFuncName,
																pushSlotArgTypes,
																PUSHSLOT_PARAMS_ARRAY_SIZE,
																args, nulls);


					NeurDBInferenceResult *result = (NeurDBInferenceResult *) DatumGetPointer(pushSlotRes.value);

					parse_result_to_cache(result, predictstate, false);

					predictstate->is_float = result->typeoid == FLOAT8OID;
					predictstate->id_class_map = result->id_class_map;

					predictstate->nrpstate = NEURDBPREDICT_INFERENCE_RETURN;
				}
				break;

			case NEURDBPREDICT_INFERENCE_RETURN:
				{
					if (dclist_is_empty(&predictstate->result_cache))
					{
						if (predictstate->is_final)
						{
							/* is the last batch */
							predictstate->nrpstate = NEURDBPREDICT_INFERENCE_END;
							continue;
						}
						else
						{
							/* reset slot cache */
							reset_slot_cache(predictstate);
							predictstate->num_consumed = 0;

							/* go back to retrieve the next batch */
							predictstate->nrpstate = NEURDBPREDICT_INFERENCE_COLLECT;
							continue;
						}
					}

					/* get the next slot from the cache */
					slot = predictstate->slot_cache[predictstate->num_consumed];

					/* project the slot */
					predictstate->ps.ps_ExprContext->ecxt_outertuple = slot;
					slot = ExecProject(predictstate->ps.ps_ProjInfo);

					/* get the next result from the cache */
					result_node *node = (result_node *) dclist_pop_head_node(&predictstate->result_cache);

					/* build the returning slot */
					build_result_slot(node->value, predictstate->is_float, predictstate->id_class_map, slot);

					predictstate->num_consumed += 1;
					return slot;
				}
				break;

			case NEURDBPREDICT_INFERENCE_END:
				return NULL;

			default:
				elog(ERROR, "unrecognized NeurDBPredictStateCond: %d", predictstate->nrpstate);
				return NULL;
		}
	}
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
		TargetEntry *tle = (TargetEntry *) lfirst(cell);

		if (tle == NULL || tle->resname == NULL)
		{
			elog(ERROR, "Null target column in statement");
		}
		appendStringInfo(&result, "%s", tle->resname);
		break;
	}

	result.data[result.len] = '\0';

	return result;
}

static char *
_temp_extract_model_name(NeurDBTrainOnSpec * trainOnSpec)
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

/*
 * create a string of feature columns.
 */
static StringInfoData
_temp_extract_train_on_columns(List *trainOn)
{
	StringInfoData result;

	initStringInfo(&result);

	ListCell   *cell;

	foreach(cell, trainOn)
	{
		TargetEntry *column = (TargetEntry *) lfirst(cell);

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

#if 0

	/*
	 * Initialize result tuple slot
	 */
	ExecInitResultTupleSlotTL(&predictstate->ps, &TTSOpsVirtual);
#endif

	/*
	 * Initialize result tuple slot with FIXED descriptor Need to determine
	 * upfront if we're doing classification or regression
	 */
	TupleDesc	resultDesc;
	int			natts = list_length(node->predictTargetList);

	/* Determine if we need the debug column (for classification) */
	/* You may need to determine this from node->stmt->kind or other metadata */
	bool		needsDebugColumn = (node->stmt->kind == PREDICT_CLASS);

	if (needsDebugColumn)
	{
		/* Classification: add debug column */
		resultDesc = CreateTemplateTupleDesc(natts + 1);

		int			i = 1;
		ListCell   *lc;

		foreach(lc, node->predictTargetList)
		{
			TargetEntry *tle = (TargetEntry *) lfirst(lc);

			TupleDescInitEntry(resultDesc,
							   (AttrNumber) i,
							   tle->resname,
							   exprType((Node *) tle->expr),
							   exprTypmod((Node *) tle->expr),
							   0);
			i++;
		}

		/* Add debug column */
		TupleDescInitEntry(resultDesc,
						   (AttrNumber) (natts + 1),
						   "_dbg_value",
						   FLOAT8OID,
						   -1,
						   0);
	}
	else
	{
		/* Regression: no debug column */
		resultDesc = ExecTypeFromTL(node->predictTargetList);
	}

	resultDesc = BlessTupleDesc(resultDesc);
	ExecInitResultTupleSlotTL(&predictstate->ps, &TTSOpsVirtual);
	predictstate->ps.ps_ResultTupleSlot = MakeSingleTupleTableSlot(resultDesc, &TTSOpsVirtual);
	predictstate->ps.ps_ResultTupleDesc = resultDesc;

	/*
	 * initialize projection info
	 */
	predictstate->ps.ps_ProjInfo =
		ExecBuildProjectionInfo(node->predictTargetList,
								predictstate->ps.ps_ExprContext,
								predictstate->ps.ps_ResultTupleSlot,
								(PlanState *) predictstate,
								ExecTypeFromTL(node->predictTargetList));

	StringInfoData targetColumn = construct_target_columns(node->predictTargetList);

	Datum		args[INIT_PARAMS_ARRAY_SIZE];
	bool		nulls[INIT_PARAMS_ARRAY_SIZE] = {false};

	char	   *table = _temp_extract_table_name(predictstate->stmt->fromClause);
	char	   *model = _temp_extract_model_name(predictstate->stmt->trainOnSpec);
	StringInfoData trainOnColumns = _temp_extract_train_on_columns(node->trainOn);

	ArrayType  *trainColumnArray = get_train_columns_array(table, targetColumn.data, trainOnColumns.data);

	args[0] = CStringGetTextDatum(model);
	args[1] = CStringGetTextDatum(table);
	args[2] = Int32GetDatum(NrTaskBatchSize);
	args[3] = Int32GetDatum(NrTaskEpoch);
	args[4] = Int32GetDatum(NrTaskNumBatches);
	args[5] = Int32GetDatum(NrTaskMaxFeatures);
	args[6] = PointerGetDatum(trainColumnArray);
	args[7] = CStringGetTextDatum(targetColumn.data);
	args[8] = Int32GetDatum(predictstate->stmt->kind);
	args[9] = PointerGetDatum(ExecTypeFromTL(outerPlan->targetlist));

	UdfResult	initRes = call_udf_function(initFuncName,
											initArgTypes,
											INIT_PARAMS_ARRAY_SIZE,
											args, nulls);

	if (!initRes.isnull)
	{
		bool		is_inference = DatumGetBool(initRes.value);

		if (is_inference)
		{
			predictstate->nrpstate = NEURDBPREDICT_INFERENCE_COLLECT;
		}
		else
		{
			predictstate->nrpstate = NEURDBPREDICT_TRAIN_COLLECT;
		}
	}

	/* initialize caches */
	predictstate->slot_cache = palloc(sizeof(TupleTableSlot *) * NrTaskBatchSize);
	predictstate->slot_cache_size = 0;
	predictstate->num_consumed = 0;
	dclist_init(&predictstate->result_cache);

	predictstate->curr_epoch = 0;

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
	_call_pipeline_close();
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
