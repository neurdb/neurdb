#include "postgres.h"

#include "neurdb/guc.h"

#include "executor/executor.h"
#include "executor/spi.h"
#include "nodes/execnodes.h"

#include "access/genam.h"
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

#include "util.h"

static char *initFuncName = "nr_pipeline_init";
#define INIT_PARAMS_ARRAY_SIZE 10
static Oid	initArgTypes[INIT_PARAMS_ARRAY_SIZE] =
{
	TEXTOID, //model name
	TEXTOID, //table name
	INT4OID, //batch size
	INT4OID, //epoch(number of training epochs)
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


/* ------------------------ Result Cache Linked List ------------------------ */

typedef struct result_node
{
	dlist_node	node;
	double		value;
}			result_node;

static void
push_result_to_cache(dclist_head *head, double value)
{
	result_node *node = palloc(sizeof(result_node));

	node->value = value;
	dclist_push_tail(head, &node->node);
}

static double
pop_result_from_cache(dclist_head *head)
{
	result_node *node = (result_node *) dclist_pop_head_node(head);

	return node->value;
}

/* ------------------------ Result Cache Linked List End ------------------------ */


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
				push_result_to_cache(&pstate->result_cache, value);

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
		push_result_to_cache(&pstate->result_cache, value);
	}
}


static char **
get_column_names(const char *schema_name,
				 const char *table_name,
				 const char *exclude,
				 int *num_included_out)
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

/**
 * Fill the given slot with one tuple derived from a NeurDBInferenceResult.
 * The slot already has a valid TupleDesc matching the node’s output schema.
 * Returns the same slot pointer.
 */
static TupleTableSlot *
build_result_slot(double value,
				  bool is_float,
				  List *id_class_map,
				  TupleTableSlot *slot)
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

static TupleTableSlot *
append_primary_values_to_tuple_slot(TupleTableSlot *proj_slot,
									AttrNumber *primkeyindexes,
									int nkeys,
									TupleTableSlot *slot_copy,
									int start_i)
{
	for (int i = 0; i < nkeys; i++)
	{
		Datum		value;
		bool		isNull;

		value = slot_copy->tts_values[primkeyindexes[i]];
		isNull = slot_copy->tts_isnull[primkeyindexes[i]];

		proj_slot->tts_values[start_i + i] = value;
		proj_slot->tts_isnull[start_i + i] = isNull;
	}

	return proj_slot;
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

					TupleTableSlot *slot_copy;

					if (predictstate->stmt->withPrimaryKey)
					{
						/* copy the primary key to a new slot */
						slot_copy = MakeTupleTableSlot(slot->tts_tupleDescriptor, &TTSOpsVirtual);
						ExecCopySlot(slot_copy, slot);
					}

					/* project the slot */
					predictstate->ps.ps_ExprContext->ecxt_outertuple = slot;
					slot = ExecProject(predictstate->ps.ps_ProjInfo);

					/* get the next result from the cache */
					double		value = pop_result_from_cache(&predictstate->result_cache);

					/* build the returning slot */
					build_result_slot(value, predictstate->is_float, predictstate->id_class_map, slot);

					if (predictstate->stmt->withPrimaryKey)
					{
						slot = append_primary_values_to_tuple_slot(slot,
																   predictstate->primkeyindexes,
																   predictstate->nkeys,
																   slot_copy,
																   predictstate->is_float ? 1 : 2);

						ReleaseTupleDesc(slot_copy->tts_tupleDescriptor);
					}

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
build_train_columns_array(const char *table,
						  const char *targetColumns,
						  const char *trainColumns)
{
	List	   *trainColumnsList = split_comma_c_string(trainColumns);
	Datum	   *trainColumnDatums;
	int			nTrainColumns = 0;

	if (strlen(trainColumns) == 0)
	{
		/* all columns are used for training */
		/* get all column names */
		char	  **allColumns = get_column_names("public", table, targetColumns, &nTrainColumns);

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
build_target_columns(List *targetList)
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


static TupleDesc
_copy_tuple_desc(List *targetList, TupleDesc resultDesc)
{
	int			i = 1;
	ListCell   *lc;

	foreach(lc, targetList)
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

	return resultDesc;
}

static TupleDesc
_append_primary_key_tuple_desc(TupleDesc resultDesc,
							   Relation rel,
							   int nkeys,
							   AttrNumber *keys,
							   List *predictTargetList,
							   int start_i)
{
	/* get tuple desc of the table */
	TupleDesc	tableDesc;

	tableDesc = RelationGetDescr(rel);

	for (int k = 0; k < nkeys; k++)
	{
		Form_pg_attribute attr;

		attr = TupleDescAttr(tableDesc, keys[k]);

		const char *name = NameStr(attr->attname);

		TupleDescInitEntry(resultDesc,
						   (AttrNumber) (start_i),
						   name,
						   attr->atttypid,
						   attr->atttypmod,
						   attr->attcollation);

		start_i++;
	}

	return resultDesc;
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
	 * Initialize result tuple slot with FIXED descriptor Need to determine
	 * upfront if we're doing classification or regression
	 */
	TupleDesc	resultDesc;
	int			natts = list_length(node->predictTargetList);

	char	   *table = _temp_extract_table_name(predictstate->stmt->fromClause);

	bool		add_primary_key = node->stmt->withPrimaryKey;
	bool		use_full = false;
	int			nkeys = 0;
	AttrNumber *keys = NULL;

	int			i;

	RangeVar   *rv = makeRangeVar(NULL, table, -1);
	Oid			relid = RangeVarGetRelid(rv, NoLock, false);
	Relation	rel = relation_open(relid, AccessShareLock);

	if (add_primary_key)
	{
		Oid			ident_index;

		ident_index = RelationGetReplicaIndex(rel);

		if (OidIsValid(ident_index))
		{
			Relation	idxrel;
			Form_pg_index idx;

			idxrel = index_open(ident_index, AccessShareLock);
			idx = idxrel->rd_index;

			nkeys = idx->indnatts;
			keys = (AttrNumber *) palloc(sizeof(AttrNumber) * nkeys);

			for (i = 0; i < nkeys; i++)
			{
				if (idx->indkey.values[i] == 0)
					ereport(ERROR,
							(errmsg("replica identity index contains expressions")));

				keys[i] = idx->indkey.values[i];
			}

			index_close(idxrel, AccessShareLock);
		}
		else
		{
			/* no primary key, use full table */
			TupleDesc	tableDesc;
			Form_pg_attribute attr;
			bool		skip;

			tableDesc = RelationGetDescr(rel);

			nkeys = tableDesc->natts - natts;
			keys = (AttrNumber *) palloc(sizeof(AttrNumber) * nkeys);

			i = 0;

			for (int k = 0; k < tableDesc->natts; k++)
			{
				attr = TupleDescAttr(tableDesc, k);

				const char *name = NameStr(attr->attname);

				/* check if the name is in the predict target list */
				ListCell   *lc;

				skip = false;
				foreach(lc, node->predictTargetList)
				{
					TargetEntry *tle = (TargetEntry *) lfirst(lc);

					if (strcmp(name, tle->resname) == 0)
					{
						/* skip this attribute */
						skip = true;
						break;
					}
				}
				if (skip)
				{
					continue;
				}

				keys[i] = k;
				i++;
			}
		}
	}

	/* Determine if we need the debug column (for classification) */
	/* You may need to determine this from node->stmt->kind or other metadata */
	bool		needsDebugColumn = (node->stmt->kind == PREDICT_CLASS);

	if (add_primary_key)
	{
		natts += nkeys;
	}

	if (needsDebugColumn)
	{
		natts++;
	}

	resultDesc = CreateTemplateTupleDesc(natts);
	resultDesc = _copy_tuple_desc(node->predictTargetList, resultDesc);

	i = list_length(node->predictTargetList) + 1;

	if (needsDebugColumn)
	{
		/* Add debug column */
		TupleDescInitEntry(resultDesc,
						   (AttrNumber) (i),
						   "_dbg_value",
						   FLOAT8OID,
						   -1,
						   0);
		i++;
	}

	if (add_primary_key)
	{
		resultDesc = _append_primary_key_tuple_desc(resultDesc,
													rel,
													nkeys,
													keys,
													node->predictTargetList,
													i);

		predictstate->primkeyindexes = keys;
		predictstate->nkeys = nkeys;
	}

	relation_close(rel, AccessShareLock);

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

	StringInfoData targetColumns = build_target_columns(node->predictTargetList);

	Datum		args[INIT_PARAMS_ARRAY_SIZE];
	bool		nulls[INIT_PARAMS_ARRAY_SIZE] = {false};

	char	   *model = _temp_extract_model_name(predictstate->stmt->trainOnSpec);
	StringInfoData trainOnColumns = _temp_extract_train_on_columns(node->trainOn);

	ArrayType  *trainColumnArray = build_train_columns_array(table,
															 targetColumns.data,
															 trainOnColumns.data);

	args[0] = CStringGetTextDatum(model);
	args[1] = CStringGetTextDatum(table);
	args[2] = Int32GetDatum(NrTaskBatchSize);
	args[3] = Int32GetDatum(NrTaskEpoch);
	args[4] = Int32GetDatum(NrTaskNumBatches);
	args[5] = Int32GetDatum(NrTaskMaxFeatures);
	args[6] = PointerGetDatum(trainColumnArray);
	args[7] = CStringGetTextDatum(targetColumns.data);
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


static void
call_pipeline_close()
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
	call_pipeline_close();
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
