#include "interface.h"

#include <unistd.h>
#include <utils/builtins.h>
#include <utils/array.h>
#include <executor/spi.h>
#include <math.h>

#include "labeling/encode.h"
#include "utils/network/http.h"
#include "utils/network/socketio.h"
#include "utils/network/socketio_nr.h"
#include "utils/metric/time_metric.h"

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
    ArrayType *features = PG_GETARG_ARRAYTYPE_P(4);
    int n_features;
    char **feature_names = text_array2char_array(features, &n_features); // column names

    // init SocketIO
    SocketIOClient *sio_client = socketio_client();

    BatchDataQueue *queue = malloc(sizeof(BatchDataQueue *));
    init_batch_data_queue(queue, 10);

    socketio_set_queue(sio_client, queue);

    socketio_register_callback(sio_client, "connection", nr_socketio_connect_callback);
    socketio_register_callback(sio_client, "request_data", nr_socketio_request_data_callback);
    socketio_connect(sio_client, "http://localhost:8090");

    while (socketio_get_socket_id(sio_client) == 0) {
        // wait for the connection
        usleep(10000); // sleep for 10ms
    }

    // prepare the query
    StringInfoData query;
    initStringInfo(&query);

    appendStringInfo(&query, "SELECT COUNT(*) FROM %s", table_name);
    // calling SPI execution
    SPI_connect();
    SPI_execute(query.data, false, 0);

    bool isnull;
    const Datum n_rows = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);

    const int n_batches = (DatumGetInt32(n_rows) - 1) / batch_size + 1; // ceil(n_rows / batch_size)

    // init dataset
    nr_socketio_emit_db_init(sio_client, table_name, n_features + 1, n_features, n_batches, 80);

    // create a new thread to send the training task
    pthread_t inference_thread;
    InferenceInfo *inference_info = malloc(sizeof(InferenceInfo));
    inference_info->model_name = model_name;
    inference_info->model_id = model_id;
    inference_info->table_name = table_name;
    inference_info->client_socket_id = socketio_get_socket_id(sio_client);
    inference_info->batch_size = batch_size;
    inference_info->batch_num = n_batches;
    pthread_create(&inference_thread, NULL, send_inference_task, (void *)inference_info);

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

    while (true) {
        SPI_execute(query.data, false, batch_size);
        if (SPI_processed == 0) {
            elog(INFO, "Inference completed");
            break; // no more rows to fetch, break the loop
        }
        resetStringInfo(&libsvm_data);
        resetStringInfo(&row_data);
        // get the query result
        SPITupleTable *tuptable = SPI_tuptable;
        TupleDesc tupdesc = tuptable->tupdesc;

        bool is_null;
        // build libsvm format data
        for (int i = 0; i < SPI_processed; i++) {
            appendStringInfoString(&row_data, "0"); // for inference, the label is always 0
            // add features
            for (int col = 0; col < n_features; col++) {
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
                        int token = encode_text(text, table_name, feature_names[col]);
                        appendStringInfo(&row_data, " %d:%d", col + 1, token);
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
        // send inference request to the Python Server
        nr_socketio_emit_batch_data(sio_client, table_name, libsvm_data.data);
        record_operation_end_time(time_metric); // record the end time of operation
        record_query_start_time(time_metric);
        resetStringInfo(&row_data);
    }
    record_query_end_time(time_metric); // record the eventual end time of query

    while (true) {}

    // clean up
    pfree(table_name);
    pfree(features);
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
 * Train the model
 * @param model_name int The name of the model to be trained
 * @param table_name text The name of the table to be used in the training
 * @param batch_size int The batch size of the input data
 * @param epochs int The number of epochs
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
    int epoch = PG_GETARG_INT32(3); // epoch
    ArrayType *features = PG_GETARG_ARRAYTYPE_P(4);
    int n_features;
    char **feature_names = text_array2char_array(features, &n_features); // feature names
    char *target = text_to_cstring(PG_GETARG_TEXT_P(5)); // target column

    // init SocketIO
    SocketIOClient *sio_client = socketio_client();

    BatchDataQueue *queue = malloc(sizeof(BatchDataQueue *));
    init_batch_data_queue(queue, 10);

    socketio_set_queue(sio_client, queue);

    socketio_register_callback(sio_client, "connection", nr_socketio_connect_callback);
    socketio_register_callback(sio_client, "request_data", nr_socketio_request_data_callback);
    socketio_connect(sio_client, "http://localhost:8090");

    while (socketio_get_socket_id(sio_client) == 0) {
        // wait for the connection
        usleep(10000); // sleep for 10ms
    }

    // prepare the query
    StringInfoData query;
    initStringInfo(&query);

    appendStringInfo(&query, "SELECT COUNT(*) FROM %s", table_name);
    // calling SPI execution
    SPI_connect();
    SPI_execute(query.data, false, 0);

    bool isnull;
    const Datum n_rows = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);

    const int n_batches = (DatumGetInt32(n_rows) - 1) / batch_size + 1; // ceil(n_rows / batch_size)
    const int n_batches_train = (int) ceil(n_batches * 0.8);
    const int n_batches_evaluate = (int) ceil(n_batches * 0.1);
    const int n_batches_test = n_batches - n_batches_train - n_batches_evaluate;

    // init dataset
    nr_socketio_emit_db_init(sio_client, table_name, n_features + 1, n_features, n_batches * epoch, 80);

    // create a new thread to send the training task
    pthread_t train_thread;
    TrainingInfo *training_info = malloc(sizeof(TrainingInfo));
    training_info->model_name = model_name;
    training_info->table_name = table_name;
    training_info->client_socket_id = socketio_get_socket_id(sio_client);
    training_info->batch_size = batch_size;
    training_info->epoch = epoch;
    training_info->train_batch_num = n_batches_train;
    training_info->eva_batch_num = n_batches_evaluate;
    training_info->test_batch_num = n_batches_test;
    pthread_create(&train_thread, NULL, send_train_task, (void *)training_info);

    // send_train_task(
    //     model_name,
    //     table_name,
    //     socketio_get_socket_id(sio_client),
    //     batch_size,
    //     epoch,
    //     n_batches_train,
    //     n_batches_evaluate,
    //     n_batches_test
    // );

    resetStringInfo(&query);
    char *cursor_name = "nr_train_cursor";
    appendStringInfo(&query, "DECLARE %s SCROLL CURSOR FOR ", cursor_name); // SCROLL is necessrary for rewind
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
        SPI_execute(query.data, false, batch_size);
        if (SPI_processed == 0) {
            current_epoch++;
            current_batch = 0;
            // if the current epoch is greater than the specified epoch, break the loop
            if (current_epoch >= epoch) {
                elog(INFO, "Training completed");
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
            appendStringInfoString(&row_data, "\n");
        }
        appendStringInfoString(&libsvm_data, row_data.data);
        record_query_end_time(time_metric);
        record_operation_start_time(time_metric);
        // send training data to the Python Server, blocking operation if the queue is full
        nr_socketio_emit_batch_data(sio_client, table_name, libsvm_data.data);
        record_operation_end_time(time_metric); // record the end time of operation
        record_query_start_time(time_metric);
        resetStringInfo(&row_data);
        current_batch++;
    }
    record_query_end_time(time_metric); // record the eventual end time of query

    while (true) {}

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
