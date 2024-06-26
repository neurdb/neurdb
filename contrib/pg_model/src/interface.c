#include "interface.h"

#include <unistd.h>
#include <catalog/pg_type_d.h>
#include <executor/spi.h>
#include <utils/array.h>
#include <utils/builtins.h>
#include "funcapi.h"

#include "access/model_sl.h"
#include "inference/model_inference.h"
#include "utils/log/logger.h"
#include "utils/torch/device.h"


PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(pgm_register_model);

PG_FUNCTION_INFO_V1(pgm_unregister_model);

PG_FUNCTION_INFO_V1(pgm_predict_float4);

PG_FUNCTION_INFO_V1(pgm_get_model_id_by_name);

PG_FUNCTION_INFO_V1(pgm_store_model);

PG_FUNCTION_INFO_V1(pgm_predict_table);

/******** PG Module Initialization ********/

/**
 * _PG_init will be called once the library is loaded by PostgreSQL, here we initialize the device (CPU or GPU)
 * @see https://www.postgresql.org/docs/current/xfunc-c.html
 */
void
_PG_init(void) {
    initialize_device();
    if (device_is_cuda()) {
        elog(INFO, "CUDA is available, using GPU for pg-model inference");
    } else {
        elog(INFO, "CUDA is not available, using CPU for pg-model inference");
    }
}

/******** Private APIs ********/

/**
 * PredictionResultData is used to stored data returned by predicting methods
 * @member {float*} data - the returned data
 * @member {long*} dims - the dimensions
 * @member {long} n_dims - the number of dimensions
 */
typedef struct {
    float *data;
    long *dims;
    long n_dims;
} PredictionResultData;

/******** Public API ********/

/*
 * register a model to the model table
 * @param model_name: the name of the model
 * @param model_path: the path of the model file (.pt)
 * @return true if success, false otherwise
 */
Datum
pgm_register_model(PG_FUNCTION_ARGS) {
    // check the number of arguments
    if (PG_NARGS() != 2) {
        ereport(ERROR, (errmsg("pgm_register_model: %d arguments are required, but only %d provided", 2, PG_NARGS())));
        PG_RETURN_BOOL(false);
    }

    const char *model_name = NULL;
    const char *model_path = NULL;

    model_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    model_path = text_to_cstring(PG_GETARG_TEXT_PP(1));

    // validate the model name and model path
    if (strlen(model_name) == 0) {
        ereport(ERROR, (errmsg("pgm_register_model: model name is empty")));
    }
    if (access(model_path, R_OK) != 0) {
        ereport(ERROR, (errmsg("pgm_register_model: model path %s is not accessible", model_path)));
    }
    // register the model
    const bool success = register_model(model_name, model_path);
    PG_RETURN_BOOL(success);
}

/*
 * unregister a model from the model table, note that the model file will not be deleted
 * @param model_name: the name of the model
 * @return true if success, false otherwise
 */
Datum
pgm_unregister_model(PG_FUNCTION_ARGS) {
    // check the number of arguments
    if (PG_NARGS() != 1) {
        ereport(ERROR,
                (errmsg("pgm_unregister_model: %d arguments are required, but only %d provided", 1, PG_NARGS())));
        PG_RETURN_BOOL(false);
    }

    const char *model_name = NULL;
    model_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    // validate the model name
    if (strlen(model_name) == 0) {
        ereport(ERROR, (errmsg("pgm_unregister_model: model name is empty")));
    }
    // drop the model
    const bool success = unregister_model(model_name);
    PG_RETURN_BOOL(success);
}

/*
 * make a prediction using a model
 * @param model_name: the name of the model
 * @param input data: the input data
 * in sql, the input is passed as an anyarray type value
 * @return prediction result in float4[]
 */
Datum
pgm_predict_float4(PG_FUNCTION_ARGS) {
    FuncCallContext *funcctx; // function calling context
    PredictionResultData *prediction_result_data; // to store the values returned

    if (SRF_IS_FIRSTCALL()) {
        funcctx = SRF_FIRSTCALL_INIT();
        MemoryContext oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

        const text *model_name_text = PG_GETARG_TEXT_PP(0);
        char *model_name = text_to_cstring(model_name_text);
        ArrayType *input_array = PG_GETARG_ARRAYTYPE_P(1);

        // check value types in the input array
        if (ARR_ELEMTYPE(input_array) != FLOAT4OID) {
            ereport(ERROR,
                    (errcode(ERRCODE_ARRAY_ELEMENT_ERROR),
                        errmsg("pgm_predict_float4:input array must be of type float4, but it is of type %s",
                            format_type_be(ARR_ELEMTYPE(input_array)))));
        }

        ModelWrapper *model = load_model_by_name(model_name); // loading model
        if (model == NULL) {
            // the model is not found
            ereport(ERROR, (errmsg("pgm_predict_float4: model %s not found", model_name)));
            PG_RETURN_NULL();
        }

        float *data = (float *) ARR_DATA_PTR(input_array); // TODO: currently only support float data
        TensorWrapper *input = tw_create_tensor(data, ARR_DIMS(input_array), ARR_NDIM(input_array));
        const TensorWrapper *output = forward(model, input); // get the prediction result

        // construct prediction result data
        prediction_result_data = (PredictionResultData *) palloc(sizeof(PredictionResultData));
        prediction_result_data->data = tw_get_tensor_data(output);
        prediction_result_data->dims = tw_get_tensor_dims(output);
        prediction_result_data->n_dims = tw_get_tensor_n_dim(output);

        // save the result and count to the calling context
        if (prediction_result_data->n_dims == 1) {
            // one sample forwarding, max call set to one
            funcctx->max_calls = 1;
        } else {
            // batch forwarding, max call set to the number of samples input
            funcctx->max_calls = prediction_result_data->dims[0];
        }
        funcctx->user_fctx = prediction_result_data;

        // number of columns equals to dims[the last dimension]
        const int column_num = (int) prediction_result_data->dims[prediction_result_data->n_dims - 1];

        TupleDesc tupdesc = CreateTemplateTupleDesc(column_num);
        for (int i = 0; i < column_num; i++) {
            char column_name[32];
            sprintf(column_name, "output_%d", i + 1);
            TupleDescInitEntry(tupdesc, (AttrNumber) (i + 1), column_name, FLOAT4OID, -1, 0);
        }
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);

        // free the model and tensors
        tw_free_model(model);
        tw_free_tensor(input);
        pfree(model_name);

        MemoryContextSwitchTo(oldcontext); // switch back to the old context
    }

    // return the result
    funcctx = SRF_PERCALL_SETUP();
    const unsigned long call_cntr = funcctx->call_cntr;
    const unsigned long max_calls = funcctx->max_calls;
    prediction_result_data = (PredictionResultData *) funcctx->user_fctx;

    const float *result_data = prediction_result_data->data;
    const long *result_dims = prediction_result_data->dims;
    const long column_num = result_dims[prediction_result_data->n_dims - 1];

    if (call_cntr < max_calls) {
        Datum values[column_num];
        bool nulls[column_num];
        memset(nulls, false, column_num * sizeof(bool));

        for (int i = 0; i < column_num; i++) {
            values[i] = Float4GetDatum(result_data[call_cntr * column_num + i]);
        }

        HeapTuple tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    } else {
        // clean up and return
        SRF_RETURN_DONE(funcctx);
    }
}

/**
 * store a model to the model table
 * @param model_name: the name of the model
 * @param model_path: the path of the model file (.pt)
 * @return true if success, false otherwise
 */
Datum
pgm_store_model(PG_FUNCTION_ARGS) {
    if (PG_NARGS() != 2) {
        ereport(ERROR, (errmsg("pgm_store_model: %d arguments are required, but only %d provided", 2, PG_NARGS())));
        PG_RETURN_BOOL(false);
    }

    const char *model_name = NULL;
    const char *model_path = NULL;
    model_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    model_path = text_to_cstring(PG_GETARG_TEXT_PP(1));

    const ModelWrapper *model = tw_load_model_by_path(model_path); // this guarantees that the model file is accessible
    const bool success = store_model(model_name, model);

    PG_RETURN_BOOL(success);
}

/*
 * get the model id by the model name
 * @param model_name: the name of the model
 * @return model id
 */
Datum
pgm_get_model_id_by_name(PG_FUNCTION_ARGS) {
    const char *model_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    const int model_id = get_model_id_by_name(model_name);
    PG_RETURN_INT32(model_id);
}

/**
 * @description make a prediction using the model by passing a table
 * @param {cstring}    model name
 * @param {int}        batch size
 * @param {cstring}    table name
 * @param {text[]}     column names
 * @return {null}      void
 * This is a temporary function designed for testing purposes,
 * it simply interacts with the SPI interface to get the data
 * from the table, and make a prediction using the same model
 * as the pgm_predict_float4 function
 *
 * It returns NULL because this function is not for real use
 */
Datum
pgm_predict_table(PG_FUNCTION_ARGS) {
    Logger logger;
    logger_init(&logger, 10);   // init the logger for testing
    logger_start(&logger, "fetching data from table");

    const char *model_name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    const int batch_size = PG_GETARG_INT32(1);
    const char *table_name = text_to_cstring(PG_GETARG_TEXT_PP(2));
    // ArrayType *column_names_array = PG_GETARG_ARRAYTYPE_P(3);
    const char *column_names_array = text_to_cstring(PG_GETARG_TEXT_PP(3));

    // create a dummy column names array for testing
    // TODO: replace this with the actual column names array
    column_names_array = "sepal_l,sepal_w,petal_l,petal_w";
    // spit by comma is the number of columns
    int num_columns = 1;
    for (int i = 0; i < strlen(column_names_array); i++) {
        if (column_names_array[i] == ',') {
            num_columns++;
        }
    }

    // int num_columns = ARR_DIMS(column_names_array)[0];
    // Datum *column_datums;   // hold the column names after deconstruction
    // bool *column_nulls;
    // deconstruct_array(column_names_array, TEXTOID, -1, false, 'i', &column_datums, &column_nulls, &num_columns);

    // StringInfoData holds information about an extensible string.
    // @see pgslq/include/server/lib/stringinfo.h
    // StringInfoData columns;
    // initStringInfo(&columns);

    // for (int i = 0; i < num_columns; ++i) {
        // if (i > 0) {
            // appendStringInfoString(&columns, ", ");
        // }
        // appendStringInfoString(&columns, TextDatumGetCString(column_datums[i]));
    // }

    // build sql query
    StringInfoData query;
    initStringInfo(&query);
    appendStringInfo(&query, "SELECT %s FROM %s", column_names_array, table_name);

    // execute query
    SPI_connect();
    if (SPI_exec(query.data, 0) != SPI_OK_SELECT) {
        SPI_finish();
        ereport(ERROR, (errmsg("failed to execute query: %s", query.data)));
    }

    logger_end(&logger);
    logger_start(&logger, "forward inference");
    // get the result
    const SPITupleTable *tuptable = SPI_tuptable;
    const int num_rows = (int) SPI_processed;
    TupleDesc tupdesc = tuptable->tupdesc;

    // forward inference
    ModelWrapper *model = load_model_by_name(model_name);

    int batch = 0;

    for (int start = 0; start < num_rows; start += batch_size) {
        const int end = (start + batch_size > num_rows) ? num_rows : start + batch_size;
        const int current_batch_size = end - start;

        // create input tensor
        float *data = (float *) palloc(current_batch_size * num_columns * sizeof(float));
        for (int i = start; i < end; ++i) {
            HeapTuple tuple = tuptable->vals[i];
            for (int j = 0; j < num_columns; ++j) {
                bool isnull;
                data[(i - start) * num_columns + j] = DatumGetFloat4(SPI_getbinval(tuple, tupdesc, j + 1, &isnull));
            }
        }
        const int dims[2] = {current_batch_size, num_columns};
        TensorWrapper *input = tw_create_tensor(data, dims, 2);

        // forward inference
        forward(model, input);
        batch++;
        elog(INFO, "[batch %d] forward inference completed", batch);

        // clean up
        tw_free_tensor(input);
        pfree(data);
    }
    logger_end(&logger);
    SPI_finish();

    // clean up
    tw_free_model(model);
    logger_print(&logger);
    PG_RETURN_NULL();
}