#include "interface.h"

#include <utils/builtins.h>
#include <utils/array.h>
#include <executor/spi.h>

#include "labeling/encode.h"
#include "utils/network/http.h"
#include "utils/metric/time_metric.h"
#include "utils/network/websocket.h"


PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(nr_inference);

PG_FUNCTION_INFO_V1(nr_train);

PG_FUNCTION_INFO_V1(nr_finetune);


// ******** Helper functions ********
char **text_array2char_array(ArrayType *text_array, int *n_elements_out);


/**
 * Preprocess the input data for model inference. It contains the following steps:
 * 1. Extract the input data from the table
 * 2. It checks if all columns' value type are integers
 *     2.1 If not, it performs one-hot encoding on the specified columns (TODO: this is not implemented yet)
 *     2.2 If yes, it continues to the next step
 * 3. It convert the data to libsvm format, and foward the data to the python server
 * @param model_name text The name of the model
 * @param model_id integer The id of the model to be used in the inference
 * @param table_name text The name of the table to be used in the inference
 * @param batch_size integer The batch size of the input data, 0 for single inference
 * @param features text[] Columns to be used in the inference
 * @return Table
 */
Datum nr_inference(PG_FUNCTION_ARGS) {
    TimeMetric *time_metric = init_time_metric("nr_inference", MILLISECOND);
    record_overall_start_time(time_metric); // record the start time of the function
    record_query_start_time(time_metric); // record the start time of preprocessing

    char *model_name = text_to_cstring(PG_GETARG_TEXT_P(0)); // model name
    int model_id = PG_GETARG_INT32(1); // model id
    char *table_name = text_to_cstring(PG_GETARG_TEXT_P(2)); // table name
    int batch_size = PG_GETARG_INT32(3); // batch size
    ArrayType *columns = PG_GETARG_ARRAYTYPE_P(4);
    int n_columns;
    char **column_names = text_array2char_array(columns, &n_columns); // column names

    // prepare the query
    StringInfoData query;
    initStringInfo(&query);

    char *cursor_name = "nr_inference_cursor";
    appendStringInfo(&query, "DECLARE %s CURSOR FOR ", cursor_name);
    appendStringInfo(&query, "SELECT ");

    for (int i = 0; i < n_columns; ++i) {
        if (i > 0) {
            appendStringInfoString(&query, ", ");
        }
        appendStringInfoString(&query, column_names[i]);
    }
    appendStringInfo(&query, " FROM %s", table_name);

    // calling SPI execution
    SPI_connect();
    SPI_execute(query.data, false, 0);

    resetStringInfo(&query);
    appendStringInfo(&query, "FETCH %d FROM %s", batch_size, cursor_name);

    StringInfoData libsvm_data;
    StringInfoData row_data;
    initStringInfo(&libsvm_data);
    initStringInfo(&row_data);

    while (true) {
        SPI_execute(query.data, false, batch_size);
        if (SPI_processed == 0) {
            break; // no more rows to fetch, break the loop
        }
        resetStringInfo(&libsvm_data);
        resetStringInfo(&row_data);
        // get the query result
        SPITupleTable *tuptable = SPI_tuptable;
        TupleDesc tupdesc = tuptable->tupdesc;

        // build libsvm format data
        for (int i = 0; i < SPI_processed; i++) {
            appendStringInfoString(&row_data, "0"); // for inference, the label is always 0
            // add features
            for (int col = 0; col < n_columns; col++) {
                bool is_null;
                Datum value = SPI_getbinval(tuptable->vals[i], tupdesc, col + 1, &is_null);
                // check the type of value
                int type = SPI_gettypeid(tupdesc, col + 1);
                switch (type) {
                    case INT2OID:
                        appendStringInfo(&row_data, " %d:%hd", col + 1, DatumGetInt16(value));
                        break;
                    case INT4OID:
                        appendStringInfo(&row_data, " %d:%d", col + 1, DatumGetInt32(value));
                        break;
                    case INT8OID:
                        appendStringInfo(&row_data, " %d:%ld", col + 1, DatumGetInt64(value));
                        break;
                    case FLOAT4OID:
                        appendStringInfo(&row_data, " %d:%f", col + 1, DatumGetFloat4(value));
                        break;
                    case FLOAT8OID:
                        appendStringInfo(&row_data, " %d:%lf", col + 1, DatumGetFloat8(value));
                        break;
                    case TEXTOID:
                    case VARCHAROID:
                    case CHAROID:
                        // do tokenization
                        char *text = DatumGetCString(value);
                        int token = encode_text(text, table_name, column_names[col]);
                        appendStringInfo(&row_data, " %d:%d", col + 1, token);
                        break;
                    default:
                        SPI_finish();
                        elog(ERROR, "Unsupported data type");
                }
            }
            appendStringInfoString(&libsvm_data, row_data.data);
            appendStringInfoChar(&libsvm_data, '\n');

            record_query_end_time(time_metric);
            record_operation_start_time(time_metric);
            // send inference request to the Python Server
            request_inference(libsvm_data.data, model_name, model_id, batch_size);
            record_operation_end_time(time_metric); // record the end time of operation
            record_query_start_time(time_metric);
            resetStringInfo(&row_data);
        }
    }
    record_query_end_time(time_metric); // record the eventual end time of query

    // clean up
    pfree(table_name);
    pfree(columns);
    for (int i = 0; i < n_columns; ++i) {
        pfree(column_names[i]);
    }
    pfree(column_names);
    SPI_finish();

    record_overall_end_time(time_metric); // record the end time of the function
    postgres_log_time(time_metric); // log the time metric
    free_time_metric(time_metric);
    PG_RETURN_NULL();
}


/**
 * Train the model
 * @param model_name int The name of the model to be trained
 * @param table_name text The name of the table to be used in the training
 * @param batch_size int The batch size of the input data
 * @param features text[] Columns to be used in the training
 * @param target text The target column
 * @return void
 */
Datum nr_train(PG_FUNCTION_ARGS) {
    TimeMetric *time_metric = init_time_metric("nr_train", MILLISECOND);
    record_overall_start_time(time_metric); // record the start time of the function
    record_query_start_time(time_metric); // record the start time of preprocessing

    char *model_name = text_to_cstring(PG_GETARG_TEXT_P(0)); // model name
    char *table_name = text_to_cstring(PG_GETARG_TEXT_P(1)); // table name
    int batch_size = PG_GETARG_INT32(2); // batch size
    ArrayType *features = PG_GETARG_ARRAYTYPE_P(3);
    int n_features;
    char **feature_names = text_array2char_array(features, &n_features); // feature names
    char *target = text_to_cstring(PG_GETARG_TEXT_P(4)); // target column

    // init websocket
    WebsocketInfo *ws_info = init_ws_connection("localhost", 8090, "/", 10);
    if (ws_info == NULL) {
        elog(ERROR, "Failed to initialize the WebSocket connection");
    }

    // start websocket thread
    if (start_ws_thread(ws_info) != 0) {
        elog(ERROR, "Failed to start the WebSocket thread");
    }

    while (strlen(ws_info->sid) == 0) {
        usleep(10000); // sleep for 10ms
    }

    send_train_task(model_name, table_name, ws_info->sid, batch_size, 1, 1, 1, 1);

    // prepare the query
    StringInfoData query;
    initStringInfo(&query);

    char *cursor_name = "nr_train_cursor";
    appendStringInfo(&query, "DECLARE %s CURSOR FOR ", cursor_name);
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
    StringInfoData row_data;
    initStringInfo(&libsvm_data);
    initStringInfo(&row_data);

    while (true) {
        SPI_execute(query.data, false, batch_size);
        if (SPI_processed == 0) {
            break; // no more rows to fetch, break the loop
        }
        resetStringInfo(&libsvm_data);
        resetStringInfo(&row_data);
        // get the query result
        SPITupleTable *tuptable = SPI_tuptable;
        TupleDesc tupdesc = tuptable->tupdesc;

        // build libsvm format data
        for (int i = 0; i < SPI_processed; i++) {
            // add label
            bool is_null;
            Datum value = SPI_getbinval(tuptable->vals[i], tupdesc, n_features + 1, &is_null);
            appendStringInfo(&row_data, "%d", DatumGetInt32(value));
            // add features
            for (int col = 0; col < n_features; col++) {
                value = SPI_getbinval(tuptable->vals[i], tupdesc, col + 1, &is_null);
                // check the type of value
                int type = SPI_gettypeid(tupdesc, col + 1);
                switch (type) {
                    case INT2OID:
                        appendStringInfo(&row_data, " %d:%hd", col + 1, DatumGetInt16(value));
                        break;
                    case INT4OID:
                        appendStringInfo(&row_data, " %d:%d", col + 1, DatumGetInt32(value));
                        break;
                    case INT8OID:
                        appendStringInfo(&row_data, " %d:%ld", col + 1, DatumGetInt64(value));
                        break;
                    case FLOAT4OID:
                        appendStringInfo(&row_data, " %d:%f", col + 1, DatumGetFloat4(value));
                        break;
                    case FLOAT8OID:
                        appendStringInfo(&row_data, " %d:%lf", col + 1, DatumGetFloat8(value));
                        break;
                    case TEXTOID:
                    case VARCHAROID:
                    case CHAROID:
                        // do tokenization
                        char *text = DatumGetCString(value);
                        int token = encode_text(text, table_name, feature_names[col]);
                        appendStringInfo(&row_data, " %d:%d", col + 1, token);
                        break;
                    default:
                        SPI_finish();
                        elog(ERROR, "Unsupported data type");
                }
            }
            appendStringInfoString(&libsvm_data, row_data.data);
            appendStringInfoChar(&libsvm_data, '\n');

            record_query_end_time(time_metric);
            record_operation_start_time(time_metric);
            // send training request to the Python Server
            send_batch_data(ws_info, table_name, TRAIN, libsvm_data.data);
            record_operation_end_time(time_metric); // record the end time of operation
            record_query_start_time(time_metric);
            resetStringInfo(&row_data);
        }
    }
    record_query_end_time(time_metric); // record the eventual end time of query

    // clean up
    pfree(table_name);
    pfree(features);
    pfree(target);
    for (int i = 0; i < n_features; ++i) {
        pfree(feature_names[i]);
    }
    pfree(feature_names);
    SPI_finish();

    record_overall_end_time(time_metric); // record the end time of the function
    postgres_log_time(time_metric); // log the time metric
    free_time_metric(time_metric);
    PG_RETURN_NULL();
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
    record_overall_start_time(time_metric); // record the start time of the function
    record_query_start_time(time_metric); // record the start time of preprocessing

    char *model_name = text_to_cstring(PG_GETARG_TEXT_P(0)); // model name
    int model_id = PG_GETARG_INT32(1); // model id
    char *table_name = text_to_cstring(PG_GETARG_TEXT_P(2)); // table name
    int batch_size = PG_GETARG_INT32(3); // batch size
    ArrayType *features = PG_GETARG_ARRAYTYPE_P(4);
    int n_features;
    char **feature_names = text_array2char_array(features, &n_features); // feature names
    char *target = text_to_cstring(PG_GETARG_TEXT_P(5)); // target column

    // prepare the query
    StringInfoData query;
    initStringInfo(&query);

    char *cursor_name = "nr_finetune_cursor";
    appendStringInfo(&query, "DECLARE %s CURSOR FOR ", cursor_name);
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
    // read_only must be set to false, otherwise FETCH will not work
    // FETCH will not work if the query is VOLATILE
    SPI_execute(query.data, false, 0);

    resetStringInfo(&query);
    appendStringInfo(&query, "FETCH %d FROM %s", batch_size, cursor_name);

    StringInfoData libsvm_data;
    StringInfoData row_data;
    initStringInfo(&libsvm_data);
    initStringInfo(&row_data);

    while (true) {
        // read_only must be set to false, otherwise FETCH will not work
        // FETCH will not work if the query is VOLATILE
        SPI_execute(query.data, false, batch_size);
        if (SPI_processed == 0) {
            break; // no more rows to fetch, break the loop
        }
        resetStringInfo(&libsvm_data);
        resetStringInfo(&row_data);
        // get the query result
        SPITupleTable *tuptable = SPI_tuptable;
        TupleDesc tupdesc = tuptable->tupdesc;

        // build libsvm format data
        for (int i = 0; i < SPI_processed; i++) {
            // add label
            bool is_null;
            Datum value = SPI_getbinval(tuptable->vals[i], tupdesc, n_features + 1, &is_null);
            appendStringInfo(&row_data, "%d", DatumGetInt32(value));
            // add features
            for (int col = 0; col < n_features; col++) {
                value = SPI_getbinval(tuptable->vals[i], tupdesc, col + 1, &is_null);
                // check the type of value
                int type = SPI_gettypeid(tupdesc, col + 1);
                switch (type) {
                    case INT2OID:
                        appendStringInfo(&row_data, " %d:%hd", col + 1, DatumGetInt16(value));
                        break;
                    case INT4OID:
                        appendStringInfo(&row_data, " %d:%d", col + 1, DatumGetInt32(value));
                        break;
                    case INT8OID:
                        appendStringInfo(&row_data, " %d:%ld", col + 1, DatumGetInt64(value));
                        break;
                    case FLOAT4OID:
                        appendStringInfo(&row_data, " %d:%f", col + 1, DatumGetFloat4(value));
                        break;
                    case FLOAT8OID:
                        appendStringInfo(&row_data, " %d:%lf", col + 1, DatumGetFloat8(value));
                        break;
                    case TEXTOID:
                    case VARCHAROID:
                    case CHAROID:
                        // do tokenization
                        char *text = DatumGetCString(value);
                        int token = encode_text(text, table_name, feature_names[col]);
                        appendStringInfo(&row_data, " %d:%d", col + 1, token);
                        break;
                    default:
                        SPI_finish();
                        elog(ERROR, "Unsupported data type");
                }
            }
            appendStringInfoString(&libsvm_data, row_data.data);
            appendStringInfoChar(&libsvm_data, '\n');

            record_query_end_time(time_metric);
            record_operation_start_time(time_metric);
            request_finetune(libsvm_data.data, model_name, model_id, batch_size); // TODO: change here
            record_operation_end_time(time_metric); // record the end time of operation
            record_query_start_time(time_metric);
            resetStringInfo(&row_data);
        }
    }
    record_query_end_time(time_metric); // record the eventual end time of query

    // clean up
    pfree(table_name);
    pfree(features);
    pfree(target);
    for (int i = 0; i < n_features; ++i) {
        pfree(feature_names[i]);
    }
    pfree(feature_names);
    SPI_finish();

    record_overall_end_time(time_metric); // record the end time of the function
    postgres_log_time(time_metric); // log the time metric
    free_time_metric(time_metric);
    PG_RETURN_NULL();
}


/**
 * Convert a text array to a char array
 * @param text_array ArrayType * The text array, usually from PG_GETARG_ARRAYTYPE_P
 * @param n_elements_out int * The number of elements in the array
 * @return char ** The char array
 */
char **text_array2char_array(ArrayType *text_array, int *n_elements_out) {
    Datum *elements;
    bool *nulls;

    deconstruct_array(text_array, TEXTOID, -1, false, 'i', &elements, &nulls, n_elements_out);
    char **char_array = (char **) palloc(*n_elements_out * sizeof(char *));
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
