#include "interface.h"

#include <utils/builtins.h>
#include <utils/array.h>
#include <executor/spi.h>
#include <math.h>

#include "labeling/encode.h"
#include "utils/metric/time_metric.h"
#include "utils/hash/md5.h"
#include "utils/network/websocket.h"

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(nr_inference);

PG_FUNCTION_INFO_V1(nr_train);

PG_FUNCTION_INFO_V1(nr_finetune);

PG_FUNCTION_INFO_V1(nr_model_lookup);

// ******** Helper functions ********
char **text_array2char_array(ArrayType *text_array, int *n_elements_out);

char *char_array2str(char **char_array, int n_elements);

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
  record_overall_start_time(
      time_metric);  // record the start time of the function
  record_query_start_time(
      time_metric);  // record the start time of preprocessing

  char *model_name = text_to_cstring(PG_GETARG_TEXT_P(0));  // model name
  int model_id = PG_GETARG_INT32(1);                        // model id
  char *table_name = text_to_cstring(PG_GETARG_TEXT_P(2));  // table name
  int batch_size = PG_GETARG_INT32(3);                      // batch size
  int batch_num = PG_GETARG_INT32(4);                       // batch number
  int nfeat = PG_GETARG_INT32(5);  // max number of input ids
  ArrayType *features = PG_GETARG_ARRAYTYPE_P(6);
  int n_features;
  char **feature_names =
      text_array2char_array(features, &n_features);  // column names

  NrWebsocket *ws = nws_initialize("localhost", 8090, "/ws", 10);
  nws_connect(ws);

  // prepare the query
  StringInfoData query;
  initStringInfo(&query);
  SPI_connect();

  const int n_batches = batch_num;

  // init dataset
  InferenceTaskSpec *inference_task_spec = malloc(sizeof(InferenceTaskSpec));
  init_inference_task_spec(inference_task_spec, model_name, batch_size,
                           n_batches, "metrics", 80, nfeat, n_features,
                           model_id);
  nws_send_task(ws, T_INFERENCE, table_name, inference_task_spec);
  free_inference_task_spec(inference_task_spec);

  resetStringInfo(&query);
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

  // create the cursor for training data
  SPI_execute(query.data, false, 0);

  resetStringInfo(&query);
  appendStringInfo(&query, "FETCH %d FROM %s", batch_size, cursor_name);

  StringInfoData libsvm_data;
  StringInfoData row_data;
  initStringInfo(&libsvm_data);
  initStringInfo(&row_data);
  int current_batch = 0;

  while (true) {
    current_batch++;
    SPI_execute(query.data, false, batch_size);
    if (SPI_processed == 0 || current_batch > n_batches) {
      elog(INFO, "Inference completed");
      break;  // no more rows to fetch, break the loop
    }
    resetStringInfo(&libsvm_data);
    resetStringInfo(&row_data);
    // get the query result
    SPITupleTable *tuptable = SPI_tuptable;
    TupleDesc tupdesc = tuptable->tupdesc;

    bool is_null;
    // build libsvm format data
    for (int i = 0; i < SPI_processed; i++) {
      appendStringInfoString(&row_data,
                             "0");  // for inference, the label is always 0
      // add features
      for (int col = 0; col < n_features; col++) {
        Datum value =
            SPI_getbinval(tuptable->vals[i], tupdesc, col + 1, &is_null);
        // check the type of value
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
            // do tokenization
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
    }
    appendStringInfoString(&libsvm_data, row_data.data);
    record_query_end_time(time_metric);
    record_operation_start_time(time_metric);
    // send training data to the Python Server, blocking operation if the queue
    // is full
    nws_send_batch_data(ws, 0, S_INFERENCE, libsvm_data.data);
    record_operation_end_time(time_metric);  // record the end time of operation
    record_query_start_time(time_metric);
    resetStringInfo(&row_data);
  }
  record_query_end_time(time_metric);  // record the eventual end time of query
  nws_wait_completion(ws);

  // clean up
  pfree(table_name);
  pfree(features);
  for (int i = 0; i < n_features; ++i) {
    pfree(feature_names[i]);
  }
  pfree(feature_names);

  // close the connection
  nws_disconnect(ws);
  nws_free_websocket(ws);

  SPI_finish();
  record_overall_end_time(time_metric);
  postgres_log_time(time_metric);  // log the time metric
  free_time_metric(time_metric);

  char *presult = pstrdup(ws->result);
  free(ws->result);

  PG_RETURN_CSTRING(presult);
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
  int batch_num = PG_GETARG_INT32(3);  // batch number for each epoch
  int epoch = PG_GETARG_INT32(4);      // epoch
  int nfeat = PG_GETARG_INT32(5);      // max number of input ids
  ArrayType *features = PG_GETARG_ARRAYTYPE_P(6);
  int n_features;
  char **feature_names =
      text_array2char_array(features, &n_features);     // feature names
  char *target = text_to_cstring(PG_GETARG_TEXT_P(7));  // target column

  NrWebsocket *ws = nws_initialize("localhost", 8090, "/ws", 10);
  nws_connect(ws);

  // prepare the query
  StringInfoData query;
  initStringInfo(&query);
  SPI_connect();

  const int n_batches = batch_num;
  const int n_batches_train = (int)ceil(n_batches * 0.8);
  const int n_batches_evaluate = (int)ceil(n_batches * 0.1);
  const int n_batches_test = n_batches - n_batches_train - n_batches_evaluate;

  // init dataset
  TrainTaskSpec *train_task_spec = malloc(sizeof(TrainTaskSpec));
  init_train_task_spec(
      train_task_spec, model_name, batch_size, epoch, n_batches_train,
      n_batches_evaluate, n_batches_test, 0.001, "optimizer", "loss", "metrics",
      80, char_array2str(feature_names, n_features), target, nfeat, n_features);
  nws_send_task(ws, T_TRAIN, table_name, train_task_spec);
  free_train_task_spec(train_task_spec);

  resetStringInfo(&query);
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

  // create the cursor for training data
  SPI_execute(query.data, false, 0);

  resetStringInfo(&query);
  appendStringInfo(&query, "FETCH %d FROM %s", batch_size, cursor_name);

  StringInfoData libsvm_data;
  StringInfoData row_data;
  initStringInfo(&libsvm_data);
  initStringInfo(&row_data);
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
    resetStringInfo(&row_data);
    // get the query result
    SPITupleTable *tuptable = SPI_tuptable;
    TupleDesc tupdesc = tuptable->tupdesc;

    bool is_null;
    // build libsvm format data
    for (int i = 0; i < SPI_processed; i++) {
      // add label
      Datum value =
          SPI_getbinval(tuptable->vals[i], tupdesc, n_features + 1, &is_null);
      appendStringInfo(&row_data, "%d", DatumGetInt32(value));
      // add features
      for (int col = 0; col < n_features; col++) {
        value = SPI_getbinval(tuptable->vals[i], tupdesc, col + 1, &is_null);
        // check the type of value
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
            // do tokenization
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
    }
    appendStringInfoString(&libsvm_data, row_data.data);
    record_query_end_time(time_metric);
    record_operation_start_time(time_metric);
    // send training data to the Python Server, blocking operation if the queue
    // is full
    nws_send_batch_data(ws, 0, S_TRAIN, libsvm_data.data);
    record_operation_end_time(time_metric);  // record the end time of operation
    record_query_start_time(time_metric);
    resetStringInfo(&row_data);
  }
  record_query_end_time(time_metric);  // record the eventual end time of query
  nws_wait_completion(ws);

  // clean up
  pfree(table_name);
  pfree(features);
  pfree(target);
  for (int i = 0; i < n_features; ++i) {
    pfree(feature_names[i]);
  }
  pfree(feature_names);

  int model_id = ws->model_id;

  // close the connection
  nws_disconnect(ws);
  nws_free_websocket(ws);

  SPI_finish();
  record_overall_end_time(time_metric);
  postgres_log_time(time_metric);  // log the time metric
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
  int batch_num = PG_GETARG_INT32(4);                       // batch number
  int epoch = PG_GETARG_INT32(5);                           // epoch
  int nfeat = PG_GETARG_INT32(6);  // max number of input ids
  ArrayType *features = PG_GETARG_ARRAYTYPE_P(7);
  int n_features;
  char **feature_names =
      text_array2char_array(features, &n_features);     // feature names
  char *target = text_to_cstring(PG_GETARG_TEXT_P(8));  // target column

  NrWebsocket *ws = nws_initialize("localhost", 8090, "/ws", 10);
  nws_connect(ws);

  // prepare the query
  StringInfoData query;
  initStringInfo(&query);
  SPI_connect();

  const int n_batches = batch_num;
  const int n_batches_train = (int)ceil(n_batches * 0.8);
  const int n_batches_evaluate = (int)ceil(n_batches * 0.1);
  const int n_batches_test = n_batches - n_batches_train - n_batches_evaluate;

  // init dataset
  FinetuneTaskSpec *finetune_task_spec = malloc(sizeof(FinetuneTaskSpec));
  init_finetune_task_spec(finetune_task_spec, model_name, model_id, batch_size,
                          epoch, n_batches_train, n_batches_evaluate,
                          n_batches_test, 0.001, "optimizer", "loss", "metrics",
                          80, nfeat, n_features);
  nws_send_task(ws, T_FINETUNE, table_name, finetune_task_spec);
  free_finetune_task_spec(finetune_task_spec);

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
  SPI_execute(query.data, false, 0);

  resetStringInfo(&query);
  appendStringInfo(&query, "FETCH %d FROM %s", batch_size, cursor_name);

  StringInfoData libsvm_data;
  StringInfoData row_data;
  initStringInfo(&libsvm_data);
  initStringInfo(&row_data);
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
        elog(INFO, "Finetune completed");
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
    resetStringInfo(&row_data);
    // get the query result
    SPITupleTable *tuptable = SPI_tuptable;
    TupleDesc tupdesc = tuptable->tupdesc;

    bool is_null;
    // build libsvm format data
    for (int i = 0; i < SPI_processed; i++) {
      // add label
      Datum value =
          SPI_getbinval(tuptable->vals[i], tupdesc, n_features + 1, &is_null);
      appendStringInfo(&row_data, "%d", DatumGetInt32(value));
      // add features
      for (int col = 0; col < n_features; col++) {
        value = SPI_getbinval(tuptable->vals[i], tupdesc, col + 1, &is_null);
        // check the type of value
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
            // do tokenization
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
    }
    appendStringInfoString(&libsvm_data, row_data.data);
    record_query_end_time(time_metric);
    record_operation_start_time(time_metric);
    nws_send_batch_data(ws, 0, S_TRAIN, libsvm_data.data);
    record_operation_end_time(time_metric);  // record the end time of operation
    record_query_start_time(time_metric);
    resetStringInfo(&row_data);
  }
  record_query_end_time(time_metric);  // record the eventual end time of query
  nws_wait_completion(ws);

  // clean up
  pfree(table_name);
  pfree(features);
  pfree(target);
  for (int i = 0; i < n_features; ++i) {
    pfree(feature_names[i]);
  }
  pfree(feature_names);

  // close the connection
  nws_disconnect(ws);
  nws_free_websocket(ws);

  SPI_finish();
  record_overall_end_time(time_metric);
  postgres_log_time(time_metric);  // log the time metric
  free_time_metric(time_metric);

  PG_RETURN_NULL();
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
