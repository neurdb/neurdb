#include "interface.h"

#include <nodes/pg_list.h>
#include <utils/builtins.h>
#include <utils/array.h>
#include <utils/memutils.h>
#include <utils/hsearch.h>
#include <executor/spi.h>
#include <math.h>
#include <neurdb/predict.h>

#include "labeling/encode.h"
#include "utils/metric/time_metric.h"
#include "utils/hash/md5.h"
#include "utils/network/websocket.h"

char *NrAIEngineHost = "localhost";
int NrAIEnginePort = 8090;

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(nr_inference);

PG_FUNCTION_INFO_V1(nr_train);

PG_FUNCTION_INFO_V1(nr_finetune);

PG_FUNCTION_INFO_V1(nr_model_lookup);

// ******** Helper functions ********
char **text_array2char_array(ArrayType *text_array, int *n_elements_out);

char *char_array2str(char **char_array, int n_elements);

void _build_libsvm_data(SPITupleTable *tuptable, TupleDesc tupdesc,
                        int n_features, char **feature_names, char *table_name,
                        StringInfo libsvm_data, bool has_label, int label_col);

static NrWebsocket *_connect_to_ai_engine() {
  NrWebsocket *ws = nws_initialize(NrAIEngineHost, NrAIEnginePort, "/ws", 10);
  nws_connect(ws);
  return ws;
}

static void _get_n_batches_train_eval_test(int n_batches, int *n_batches_train,
                                           int *n_batches_eval,
                                           int *n_batches_test) {
  *n_batches_train = (int)ceil(n_batches * 0.8);
  *n_batches_eval = (int)ceil(n_batches * 0.1);
  *n_batches_test = n_batches - *n_batches_train - *n_batches_eval;
}

static void _clean_up_common(NrWebsocket *ws, char *table_name,
                             ArrayType *features, char **feature_names,
                             int n_features) {
  // free
  pfree(table_name);
  pfree(features);
  for (int i = 0; i < n_features; ++i) {
    pfree(feature_names[i]);
  }
  pfree(feature_names);

  // close the connection
  nws_disconnect(ws);
  nws_free_websocket(ws);

  // close SPI
  SPI_finish();
}

/**
 * The last class id map. This is a temporary solution to allow inference to
 * get the class id map.
 */
static HTAB *last_class_id_map = NULL;
static List *last_id_class_map = NULL;

/**
 * @brief Make a map from class names to class id.
 *
 * TODO: store the map in the database
 * @param table_name name of the target table
 * @param label_col_name name of the label column in the target table
 * @param class_id_map (out) the map from class names to class id, using HTAB
 * @param id_class_map (out) the map from class id to class names, using List
 * @return
 */
static void make_class_id_map(const char *table_name,
                              const char *label_col_name, HTAB **class_id_map,
                              List **id_class_map) {
  StringInfoData query;
  initStringInfo(&query);

  appendStringInfo(&query, "SELECT DISTINCT %s FROM %s ORDER BY %s ASC",
                   label_col_name, table_name, label_col_name);

  SPI_execute(query.data, true, 0);

  HASHCTL ctl;
  memset(&ctl, 0, sizeof(ctl));
  ctl.keysize = sizeof(char *);
  ctl.entrysize = sizeof(int);

  HTAB *cimap =
      hash_create("neurdb class id map", 1024, &ctl, HASH_ELEM | HASH_STRINGS);
  List *icmap = NIL;

  int num_class = 0;
  bool found = 0;

  MemoryContext oldcxt;

  if (SPI_processed > 0) {
    num_class = SPI_processed;

    for (int i = 0; i < SPI_processed; i++) {
      char *label =
          SPI_getvalue(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1);

      int *id = hash_search(cimap, (void *)label, HASH_ENTER, &found);
      if (!found) {
        *id = i;
      }

      oldcxt = MemoryContextSwitchTo(TopMemoryContext);
      icmap = lappend(icmap, makeString(pstrdup(label)));
      MemoryContextSwitchTo(oldcxt);
    }
  }

  *class_id_map = cimap;
  *id_class_map = icmap;
}

/**
 * Preprocess the input data for model inference. It contains the following
 * steps:
 * 1. Extract the input data from the table
 * 2. It checks if all columns' value type are integers
 *     2.1 If not, it performs one-hot encoding on the specified columns (TODO:
 * this is not implemented yet) 2.2 If yes, it continues to the next step
 * 3. It convert the data to libsvm format, and foward the data to the python
 * server
 * @param model_name text The name of the model
 * @param model_id integer The id of the model to be used in the inference
 * @param table_name text The name of the table to be used in the inference
 * @param batch_size integer The batch size of the input data, 0 for single
 * inference
 * @param features text[] Columns to be used in the inference
 * @return Table
 */
Datum nr_inference(PG_FUNCTION_ARGS) {
  TimeMetric *time_metric = init_time_metric("nr_inference", MILLISECOND);
  // record the start time of the function
  record_overall_start_time(time_metric);
  // record the start time of preprocessing
  record_query_start_time(time_metric);

  char *model_name = text_to_cstring(PG_GETARG_TEXT_P(0));  // model name
  int model_id = PG_GETARG_INT32(1);                        // model id
  char *table_name = text_to_cstring(PG_GETARG_TEXT_P(2));  // table name
  int batch_size = PG_GETARG_INT32(3);                      // batch size
  int n_batches = PG_GETARG_INT32(4);                       // batch number
  int nfeat = PG_GETARG_INT32(5);  // max number of input ids
  ArrayType *features = PG_GETARG_ARRAYTYPE_P(6);
  PredictType type = PG_GETARG_INT32(7);

  int n_features;
  char **feature_names =
      text_array2char_array(features, &n_features);  // column names

  NrWebsocket *ws = _connect_to_ai_engine();

  SPI_connect();

  int n_class = -1;

  if (type == PREDICT_CLASS) {
    if (last_class_id_map == NULL) {
      elog(ERROR,
           "last_class_id_map is NULL. Currently, inference is not "
           "supported if training is not done within the same session. "
           "Please train the model in this session first.");
    }
    n_class = hash_get_num_entries(last_class_id_map);
    elog(DEBUG1, "n_class: %d", n_class);

    if (n_class != 2) {
      elog(ERROR,
           "only binary classification is supported for now. The label column "
           "has %d classes",
           n_class);
    }

    for (int i = 0; i < n_class; i++) {
      elog(DEBUG1, "class %d: %s", i,
           ((String *)list_nth(last_id_class_map, i))->sval);
    }
  }

  // init dataset
  InferenceTaskSpec *inference_task_spec = malloc(sizeof(InferenceTaskSpec));
  init_inference_task_spec(inference_task_spec, model_name, batch_size,
                           n_batches, "metrics", 80, nfeat, n_features, n_class,
                           model_id);
  nws_send_task(ws, T_INFERENCE, table_name, inference_task_spec);
  free_inference_task_spec(inference_task_spec);

  // prepare the query
  StringInfoData query;
  initStringInfo(&query);
  resetStringInfo(&query);

  // create the cursor for inference data
  char *cursor_name = "nr_inference_cursor";
  appendStringInfo(&query, "DECLARE %s SCROLL CURSOR FOR ", cursor_name);
  appendStringInfo(&query, "SELECT ");

  for (int i = 0; i < n_features; ++i) {
    if (i > 0) {
      appendStringInfoString(&query, ", ");
    }
    appendStringInfoString(&query, feature_names[i]);
  }
  appendStringInfo(&query, " FROM %s", table_name);

  SPI_execute(query.data, false, 0);

  resetStringInfo(&query);
  appendStringInfo(&query, "FETCH %d FROM %s", batch_size, cursor_name);

  StringInfoData libsvm_data;
  initStringInfo(&libsvm_data);
  int current_batch = 0;

  while (true) {
    current_batch++;
    SPI_execute(query.data, false, batch_size);
    if (SPI_processed == 0 || current_batch > n_batches) {
      elog(NOTICE, "Inference completed");
      break;  // no more rows to fetch, break the loop
    }

    resetStringInfo(&libsvm_data);

    // get the query result
    SPITupleTable *tuptable = SPI_tuptable;
    TupleDesc tupdesc = tuptable->tupdesc;

    _build_libsvm_data(tuptable, tupdesc, n_features, feature_names, table_name,
                       &libsvm_data, false, 0);

    record_query_end_time(time_metric);
    record_operation_start_time(time_metric);

    // send training data to the Python Server, blocking operation if the queue
    // is full
    nws_send_batch_data(ws, 0, S_INFERENCE, libsvm_data.data);

    record_operation_end_time(time_metric);  // record the end time of operation
    record_query_start_time(time_metric);
    // resetStringInfo(&row_data);
  }
  record_query_end_time(time_metric);  // record the eventual end time of query

  nws_wait_completion(ws);

  _clean_up_common(ws, table_name, features, feature_names, n_features);

  record_overall_end_time(time_metric);
  elog_time(time_metric);  // log the time metric
  free_time_metric(time_metric);

  char *presult = pstrdup(ws->result);
  free(ws->result);

  NeurDBInferenceResult *result = palloc(sizeof(NeurDBInferenceResult));
  // TODO: infer the type of the result
  if (type == PREDICT_CLASS) {
    result->typeoid = TEXTOID;
  } else if (type == PREDICT_VALUE) {
    result->typeoid = FLOAT8OID;
  } else {
    elog(ERROR, "Unsupported data type");
  }
  result->result = presult;
  result->id_class_map = last_id_class_map;

  PG_RETURN_POINTER(result);
}

/**
 * Train the model
 * @param model_name int The name of the model to be trained
 * @param table_name text The name of the table to be used in the training
 * @param batch_size int The batch size of the input data
 * @param epochs int The number of epochs
 * @param nfeat int The max number of input ids
 * @param features text[] Columns to be used in the training
 * @param target text The target column
 * @return void
 */
Datum nr_train(PG_FUNCTION_ARGS) {
  TimeMetric *time_metric = init_time_metric("nr_train", MILLISECOND);
  record_overall_start_time(time_metric);
  record_query_start_time(time_metric);

  char *model_name = text_to_cstring(PG_GETARG_TEXT_P(0));  // model name
  char *table_name = text_to_cstring(PG_GETARG_TEXT_P(1));  // table name
  int batch_size = PG_GETARG_INT32(2);                      // batch size
  int n_batches = PG_GETARG_INT32(3);  // batch number for each epoch
  int epoch = PG_GETARG_INT32(4);      // epoch
  int nfeat = PG_GETARG_INT32(5);      // max number of input ids
  ArrayType *features = PG_GETARG_ARRAYTYPE_P(6);
  char *target = text_to_cstring(PG_GETARG_TEXT_P(7));  // target column
  PredictType type = PG_GETARG_INT32(8);

  int n_features;
  char **feature_names =
      text_array2char_array(features, &n_features);  // feature names

  NrWebsocket *ws = _connect_to_ai_engine();

  int n_batches_train, n_batches_evaluate, n_batches_test;
  _get_n_batches_train_eval_test(n_batches, &n_batches_train,
                                 &n_batches_evaluate, &n_batches_test);

  SPI_connect();

  int n_class = -1;

  if (type == PREDICT_CLASS) {
    if (last_class_id_map != NULL) {
      hash_destroy(last_class_id_map);
      list_free_deep(last_id_class_map);
    }
    make_class_id_map(table_name, target, &last_class_id_map,
                      &last_id_class_map);
    n_class = hash_get_num_entries(last_class_id_map);

    elog(DEBUG1, "n_class: %d", n_class);
    for (int i = 0; i < n_class; i++) {
      elog(DEBUG1, "class %d: %s", i,
           ((String *)list_nth(last_id_class_map, i))->sval);
    }
  }

  // init dataset
  TrainTaskSpec *train_task_spec = malloc(sizeof(TrainTaskSpec));
  init_train_task_spec(train_task_spec, model_name, batch_size, epoch,
                       n_batches_train, n_batches_evaluate, n_batches_test,
                       0.001, "optimizer", "loss", "metrics", 80,
                       char_array2str(feature_names, n_features), target, nfeat,
                       n_features, n_class);
  nws_send_task(ws, T_TRAIN, table_name, train_task_spec);
  free_train_task_spec(train_task_spec);

  // prepare the query
  StringInfoData query;
  initStringInfo(&query);
  resetStringInfo(&query);

  // create the cursor for inference data
  char *cursor_name = "nr_train_cursor";
  appendStringInfo(&query, "DECLARE %s SCROLL CURSOR FOR ",
                   cursor_name);  // SCROLL is necessrary for rewind
  appendStringInfo(&query, "SELECT ");

  for (int i = 0; i < n_features; ++i) {
    if (i > 0) {
      appendStringInfoString(&query, ", ");
    }
    appendStringInfoString(&query, feature_names[i]);
  }
  appendStringInfo(&query, ", %s FROM %s", target, table_name);

  SPI_execute(query.data, false, 0);

  resetStringInfo(&query);
  appendStringInfo(&query, "FETCH %d FROM %s", batch_size, cursor_name);

  StringInfoData libsvm_data;
  initStringInfo(&libsvm_data);

  int current_epoch = 0;
  int current_batch = 0;

  while (true) {
    current_batch++;
    SPI_execute(query.data, false, batch_size);
    if (SPI_processed == 0 || current_batch > n_batches) {
      current_epoch++;
      current_batch = 1;  // reset the current batch
      // if the current epoch is greater than the specified epoch, break the
      // loop
      if (current_epoch >= epoch) {
        break;
      }
      // reset the cursor
      resetStringInfo(&query);
      appendStringInfo(&query, "MOVE ABSOLUTE 0 IN %s", cursor_name);
      SPI_execute(query.data, false, 0);

      resetStringInfo(&query);
      appendStringInfo(&query, "FETCH %d FROM %s", batch_size, cursor_name);
    }

    resetStringInfo(&libsvm_data);

    // get the query result
    SPITupleTable *tuptable = SPI_tuptable;
    TupleDesc tupdesc = tuptable->tupdesc;

    _build_libsvm_data(tuptable, tupdesc, n_features, feature_names, table_name,
                       &libsvm_data, true, n_features + 1);

    record_query_end_time(time_metric);
    record_operation_start_time(time_metric);

    // send training data to the Python Server, blocking operation if the queue
    // is full
    nws_send_batch_data(ws, 0, S_TRAIN, libsvm_data.data);

    record_operation_end_time(time_metric);  // record the end time of operation
    record_query_start_time(time_metric);
  }
  record_query_end_time(time_metric);  // record the eventual end time of query

  nws_wait_completion(ws);

  int model_id = ws->model_id;

  _clean_up_common(ws, table_name, features, feature_names, n_features);
  pfree(target);

  record_overall_end_time(time_metric);
  elog_time(time_metric);  // log the time metric
  free_time_metric(time_metric);

  PG_RETURN_INT32(model_id);
}

/**
 * Resquest the server to finetune a model
 * @param libsvm_data char* Finetune data in libsvm format
 * @param model_name char* Model name
 * @param model_id int Trained model id
 * @param batch_size int Batch size in finetune
 */
Datum nr_finetune(PG_FUNCTION_ARGS) {
  TimeMetric *time_metric = init_time_metric("nr_finetune", MILLISECOND);
  record_overall_start_time(time_metric);
  record_query_start_time(time_metric);

  char *model_name = text_to_cstring(PG_GETARG_TEXT_P(0));  // model name
  int model_id = PG_GETARG_INT32(1);                        // model id
  char *table_name = text_to_cstring(PG_GETARG_TEXT_P(2));  // table name
  int batch_size = PG_GETARG_INT32(3);                      // batch size
  int n_batches = PG_GETARG_INT32(4);                       // batch number
  int epoch = PG_GETARG_INT32(5);                           // epoch
  int nfeat = PG_GETARG_INT32(6);  // max number of input ids
  ArrayType *features = PG_GETARG_ARRAYTYPE_P(7);
  char *target = text_to_cstring(PG_GETARG_TEXT_P(8));  // target column

  int n_features;
  char **feature_names =
      text_array2char_array(features, &n_features);  // feature names

  NrWebsocket *ws = _connect_to_ai_engine();

  int n_batches_train, n_batches_evaluate, n_batches_test;
  _get_n_batches_train_eval_test(n_batches, &n_batches_train,
                                 &n_batches_evaluate, &n_batches_test);

  // init dataset
  FinetuneTaskSpec *finetune_task_spec = malloc(sizeof(FinetuneTaskSpec));
  init_finetune_task_spec(finetune_task_spec, model_name, model_id, batch_size,
                          epoch, n_batches_train, n_batches_evaluate,
                          n_batches_test, 0.001, "optimizer", "loss", "metrics",
                          80, nfeat, n_features);
  nws_send_task(ws, T_FINETUNE, table_name, finetune_task_spec);
  free_finetune_task_spec(finetune_task_spec);

  // prepare the query
  StringInfoData query;
  initStringInfo(&query);
  resetStringInfo(&query);

  char *cursor_name = "nr_finetune_cursor";
  appendStringInfo(&query, "DECLARE %s SCROLL CURSOR FOR ", cursor_name);
  appendStringInfo(&query, "SELECT ");

  for (int i = 0; i < n_features; ++i) {
    if (i > 0) {
      appendStringInfoString(&query, ", ");
    }
    appendStringInfoString(&query, feature_names[i]);
  }
  appendStringInfo(&query, ", %s FROM %s", target, table_name);

  // calling SPI execution
  SPI_connect();
  SPI_execute(query.data, false, 0);

  resetStringInfo(&query);
  appendStringInfo(&query, "FETCH %d FROM %s", batch_size, cursor_name);

  StringInfoData libsvm_data;
  initStringInfo(&libsvm_data);

  int current_epoch = 0;
  int current_batch = 0;

  while (true) {
    // read_only must be set to false, otherwise FETCH will not work
    // FETCH will not work if the query is VOLATILE
    current_batch++;
    SPI_execute(query.data, false, batch_size);
    if (SPI_processed == 0 || current_batch > n_batches) {
      current_epoch++;
      current_batch = 1;  // reset the current batch
      // if the current epoch is greater than the specified epoch, break the
      // loop
      if (current_epoch >= epoch) {
        elog(NOTICE, "Finetune completed");
        break;
      }
      // reset the cursor
      resetStringInfo(&query);
      appendStringInfo(&query, "MOVE ABSOLUTE 0 IN %s", cursor_name);
      SPI_execute(query.data, false, 0);

      resetStringInfo(&query);
      appendStringInfo(&query, "FETCH %d FROM %s", batch_size, cursor_name);
    }
    resetStringInfo(&libsvm_data);
    // get the query result
    SPITupleTable *tuptable = SPI_tuptable;
    TupleDesc tupdesc = tuptable->tupdesc;

    _build_libsvm_data(tuptable, tupdesc, n_features, feature_names, table_name,
                       &libsvm_data, true, n_features + 1);

    record_query_end_time(time_metric);
    record_operation_start_time(time_metric);

    nws_send_batch_data(ws, 0, S_TRAIN, libsvm_data.data);

    record_operation_end_time(time_metric);  // record the end time of operation
    record_query_start_time(time_metric);
  }
  record_query_end_time(time_metric);  // record the eventual end time of query

  nws_wait_completion(ws);

  _clean_up_common(ws, table_name, features, feature_names, n_features);
  pfree(target);

  record_overall_end_time(time_metric);
  elog_time(time_metric);  // log the time metric
  free_time_metric(time_metric);

  PG_RETURN_NULL();
}

/**
 * Convert a batch of SPI query results into libsvm format.
 * @param tuptable SPITupleTable* The SPI tuple table containing query results
 * @param tupdesc TupleDesc The tuple descriptor
 * @param n_features int Number of feature columns
 * @param feature_names char** Array of feature column names
 * @param table_name char* Name of the table (for tokenization)
 * @param libsvm_data StringInfo* Output buffer for libsvm-formatted data
 * @param has_label bool Whether to include a label (true for train/finetune,
 * false for inference)
 * @param label_col int Column index of the label (ignored if has_label is
 * false)
 */
void _build_libsvm_data(SPITupleTable *tuptable, TupleDesc tupdesc,
                        int n_features, char **feature_names, char *table_name,
                        StringInfo libsvm_data, bool has_label, int label_col) {
  StringInfoData row_data;
  initStringInfo(&row_data);

  bool is_null;
  for (int i = 0; i < SPI_processed; i++) {
    resetStringInfo(&row_data);

    // handle label if present
    if (has_label) {
      Datum value =
          SPI_getbinval(tuptable->vals[i], tupdesc, label_col, &is_null);
      appendStringInfo(&row_data, "%d", DatumGetInt32(value));
    } else {
      appendStringInfoString(&row_data, "0");  // Default for inference
    }

    // process features
    for (int col = 0; col < n_features; col++) {
      Datum value =
          SPI_getbinval(tuptable->vals[i], tupdesc, col + 1, &is_null);
      int type = SPI_gettypeid(tupdesc, col + 1);
      switch (type) {
        case INT2OID:
          appendStringInfo(&row_data, " %hd", DatumGetInt16(value));
          break;
        case INT4OID:
          appendStringInfo(&row_data, " %d", DatumGetInt32(value));
          break;
        case INT8OID:
          appendStringInfo(&row_data, " %ld", DatumGetInt64(value));
          break;
        case FLOAT4OID:
          appendStringInfo(&row_data, " %f", DatumGetFloat4(value));
          break;
        case FLOAT8OID:
          appendStringInfo(&row_data, " %lf", DatumGetFloat8(value));
          break;
        case TEXTOID:
        case VARCHAROID:
        case CHAROID:
          char *text = DatumGetCString(value);
          int token = encode_text(text, table_name, feature_names[col]);
          appendStringInfo(&row_data, " %d", token);
          break;
        default:
          SPI_finish();
          elog(ERROR, "Unsupported data type");
      }
    }
    appendStringInfoString(&row_data, "\n");
    appendStringInfoString(libsvm_data, row_data.data);
  }

  pfree(row_data.data);  // free the string info
}

/**
 * Check if the model exists
 * @param table_name text The name of the table to be used in the inference
 * @param features text[] Columns to be used in the inference
 * @param target text The target column
 */
Datum nr_model_lookup(PG_FUNCTION_ARGS) {
  char *table_name = text_to_cstring(PG_GETARG_TEXT_P(0));
  ArrayType *features = PG_GETARG_ARRAYTYPE_P(1);
  char *target = text_to_cstring(PG_GETARG_TEXT_P(2));

  int n_features;
  char **feature_array = text_array2char_array(features, &n_features);
  char *hash_features =
      nr_md5_list(feature_array, n_features);  // hashed features
  char *hash_target = nr_md5_str(target);      // hashed target

  StringInfoData query;
  initStringInfo(&query);

  SPI_connect();
  appendStringInfo(&query,
                   "SELECT * FROM router WHERE table_name = '%s' AND "
                   "feature_columns = '%s' AND target_columns = '%s'",
                   table_name, hash_features, hash_target);

  SPI_execute(query.data, false, 0);
  if (SPI_processed == 0) {
    SPI_finish();
    PG_RETURN_INT32(0);
  } else {
    bool is_null;
    Datum model_id = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc,
                                   1, &is_null);
    SPI_finish();
    PG_RETURN_INT32(DatumGetInt32(model_id));
  }
}

/**
 * Convert a text array to a char array
 * @param text_array ArrayType * The text array, usually from
 * PG_GETARG_ARRAYTYPE_P
 * @param n_elements_out int * The number of elements in the array
 * @return char ** The char array
 */
char **text_array2char_array(ArrayType *text_array, int *n_elements_out) {
  Datum *elements;
  bool *nulls;

  deconstruct_array(text_array, TEXTOID, -1, false, 'i', &elements, &nulls,
                    n_elements_out);
  char **char_array = (char **)palloc(*n_elements_out * sizeof(char *));
  for (int i = 0; i < *n_elements_out; i++) {
    if (nulls[i]) {
      char_array[i] = NULL;
    } else {
      char_array[i] = text_to_cstring(DatumGetTextP(elements[i]));
    }
  }
  pfree(elements);
  pfree(nulls);
  return char_array;
}

/**
 * Convert a char array into char*, separated by ','
 * @param char_array char** The char array
 * @param n_elements int The number of elements in the array
 * @return
 */
char *char_array2str(char **char_array, int n_elements) {
  StringInfoData str;
  initStringInfo(&str);
  for (int i = 0; i < n_elements; i++) {
    appendStringInfo(&str, "%s", char_array[i]);
    if (i < n_elements - 1) {
      appendStringInfoString(&str, ",");
    }
  }
  return str.data;
}
