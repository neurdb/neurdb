#include "pgext/interface.h"

#include <libpq-fe.h>
#include <catalog/namespace.h>
#include <utils/builtins.h>
#include <utils/array.h>
#include <utils/fmgroids.h>
#include <utils/lsyscache.h>
#include <sys/types.h>
#include <funcapi.h>
#include <miscadmin.h>
#include <utils/guc.h>
#include <omp.h>
#include <postmaster/bgworker.h>
#include <storage/ipc.h>
#include <storage/latch.h>

#include "neurstore/neurstore.h"
#include "neurstore/cache/index_cache_manager.h"
#include "neurstore/utils/model.h"
#include "neurstore/utils/logging.h"
#include "pgext/operation.h"
#include "pgext/callback/select.h"
#include "pgext/ipc/channel.h"
#include "pgext/ipc/service.h"
#include "pgext/global.h"


PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(ns_save_model);
PG_FUNCTION_INFO_V1(ns_save_models);
PG_FUNCTION_INFO_V1(ns_save_models_from_folder);
PG_FUNCTION_INFO_V1(ns_save_model_dry_run);
PG_FUNCTION_INFO_V1(ns_load_model);
PG_FUNCTION_INFO_V1(ns_load_model_as_uint8);
PG_FUNCTION_INFO_V1(ns_load_model_as_uint8_delta);
PG_FUNCTION_INFO_V1(ns_load_model_as_float16);
PG_FUNCTION_INFO_V1(ns_inference);
PG_FUNCTION_INFO_V1(ns_clean_cache);
// PG_FUNCTION_INFO_V1(ns_set_parallelism);
// PG_FUNCTION_INFO_V1(ns_get_parallelism);

static char STORE_PATH[MAXPGPATH] = "";
static BackgroundWorker worker;
static NSChannel *response_channel = NULL;

static shmem_request_hook_type prev_shmem_request_hook  = NULL;
static shmem_startup_hook_type prev_shmem_startup_hook  = NULL;


static void
neurstore_shmem_request(void) {
    if (prev_shmem_request_hook)
        prev_shmem_request_hook();
    RequestAddinShmemSpace(sizeof(NSChannel) * 4 * (THREAD_POOL_PARALLELISM + 1));
    RequestNamedLWLockTranche("NeurStoreChannel", 1);
}

static void
neurstore_shmem_startup(void) {
    if (prev_shmem_startup_hook)
        prev_shmem_startup_hook();
    NSChannelRegisterTranche();
    NSChannelInit(NS_CHANNEL, true);
}

static void terminate_neurstore(SIGNAL_ARGS) {
    NSChannel* request_channel = NSChannelInit(NS_CHANNEL, false);
    if (request_channel) {
        NSTaskMsg *shutdown_msg = NewTask(NS_TASK_SHUTDOWN, 0, 0, NULL);
        NSChannelPush(request_channel, shutdown_msg);
        pfree(shutdown_msg);
    }
    SetLatch(MyLatch);
}

PGDLLEXPORT void bgw_neurstore_main() {
    pqsignal(SIGTERM, terminate_neurstore);
    BackgroundWorkerUnblockSignals();
    run_neurstore(THREAD_POOL_PARALLELISM, SINGLE_THREAD_PARALLELISM, STORE_PATH);
}

void _PG_init(void) {
    memset(&worker, 0, sizeof(worker));
    prev_shmem_request_hook = shmem_request_hook;
    shmem_request_hook      = neurstore_shmem_request;
    prev_shmem_startup_hook = shmem_startup_hook;
    shmem_startup_hook      = neurstore_shmem_startup;

    strncpy(worker.bgw_name, "NeurStoreService", BGW_MAXLEN - 1);
    worker.bgw_name[BGW_MAXLEN - 1] = '\0';
    strncpy(worker.bgw_type, "CustomStorageWorker", BGW_MAXLEN - 1);
    worker.bgw_type[BGW_MAXLEN - 1] = '\0';

    worker.bgw_flags = BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
    worker.bgw_restart_time = 1;
    worker.bgw_start_time = BgWorkerStart_ConsistentState;
    snprintf(worker.bgw_library_name, BGW_MAXLEN, "pg_neurstore");
    snprintf(worker.bgw_function_name, BGW_MAXLEN, "bgw_neurstore_main");
    worker.bgw_notify_pid = 0;
    RegisterBackgroundWorker(&worker);
    const char *data_dir = GetConfigOption("data_directory", false, false);
    // TODO: hardcoded for now, move to Postgres GUC in the future
    snprintf(STORE_PATH, MAXPGPATH, "%s/neurstore_data", data_dir);
}

/**
 * Save a model to the database
 * @param model_name char* The name of the model
 * @param tolerance float The tolerance during compression
 * @param model_path char* The path to the model file (in .ONNX format)
 * @return bool True if the model is saved successfully, false otherwise
 */
Datum
ns_save_model(PG_FUNCTION_ARGS) {
    char *model_name = text_to_cstring(PG_GETARG_TEXT_P(0));
    float tolerance = PG_GETARG_FLOAT4(1);
    char *model_path = text_to_cstring(PG_GETARG_TEXT_P(2));

    if (response_channel == NULL) {
        uint32 resp_id = MyProcPid;
        char resp_name[64];
        snprintf(resp_name, sizeof resp_name, "ns_resp_%u", resp_id);
        response_channel = NSChannelInit(resp_name, true);
    }

    uint32 name_len = strlen(model_name) + 1;
    uint32 path_len = strlen(model_path) + 1;
    Size pl_size = sizeof(SaveModelPayload) + name_len + path_len;

    SaveModelPayload *payload = palloc(pl_size);
    payload->tolerance = tolerance;
    payload->name_len = name_len;
    payload->path_len = path_len;
    memcpy(payload->data, model_name, name_len);
    memcpy(payload->data + name_len, model_path, path_len);

    NSTaskMsg *msg = NewTask(NS_TASK_SAVE_MODEL, pl_size, MyProcPid, payload);

    NSChannel *req_chan = NSChannelInit(NS_CHANNEL, false);
    if (!NSChannelPush(req_chan, msg))
        ereport(ERROR, (errmsg("NeurStore: push task failed")));

    pfree(msg);
    pfree(payload);

    NSTaskMsg *response;
    for (;;) {
        response = NSChannelPop(response_channel);
        if (response != NULL)
            break;
        CHECK_FOR_INTERRUPTS();
    }

    const NSTaskMsg *reply = (const NSTaskMsg *) response;
    bool ok = (reply->type == NS_TASK_OK);
    pfree(response);

    if (!ok) {
        PG_RETURN_BOOL(false);
    }

    Relation rel = pgext_open("model", RowExclusiveLock);
    Datum nextval_result = pgext_nextval("model_model_id_seq");
    Datum values[2];
    bool nulls_insert[2] = {false, false};
    values[0] = Int32GetDatum((int32) nextval_result);
    values[1] = CStringGetTextDatum(model_name);
    pgext_insert(rel, values, nulls_insert);
    pgext_close(rel, RowExclusiveLock);
    PG_RETURN_BOOL(true);
}

/**
 * Save a model to the database
 * @param model_names char** The names of the models
 * @param tolerance float The tolerance during compression
 * @param model_folder_path char* The folder path to the model files (in .ONNX format)
 * @return bool True if the model is saved successfully, false otherwise
 */
Datum
ns_save_models(PG_FUNCTION_ARGS) {
    ArrayType *model_name_array = PG_GETARG_ARRAYTYPE_P(0);
    float tolerance = PG_GETARG_FLOAT4(1);
    char *model_folder_path = text_to_cstring(PG_GETARG_TEXT_P(2));
    if (ARR_NDIM(model_name_array) != 1) {
        LOG_FAILURE_PARAMETER_ERROR(
            "model_names must be 1-dimensional arrays, got: %d",
            ARR_NDIM(model_name_array)
        );
        PG_RETURN_BOOL(false);
    }
    int n_models = ArrayGetNItems(ARR_NDIM(model_name_array), ARR_DIMS(model_name_array));

    Datum *datum_model_names;
    bool *model_names_nulls;
    int num_models;

    deconstruct_array(
        model_name_array,
        TEXTOID,
        -1,
        false,
        'i',
        &datum_model_names,
        &model_names_nulls,
        &num_models
    );

    uint32 folder_lenth = strlen(model_folder_path) + 1;      /* include '\0' */
    Size payload_size = sizeof(SaveModelsPayload) + folder_lenth;

    for (int i = 0; i < n_models; ++i) {
        if (model_names_nulls[i]) {
            ereport(ERROR, (errmsg("model name cannot be NULL")));
            PG_RETURN_BOOL(false);
        }
        payload_size += sizeof(uint32) + strlen(TextDatumGetCString(datum_model_names[i])) + 1;
    }

    /* build up payload */
    SaveModelsPayload *payload = palloc0(payload_size);
    char *cursor = payload->data;
    payload->tolerance = tolerance;
    payload->folder_len = folder_lenth;
    payload->n_models = (uint32) n_models;

    memcpy(cursor, model_folder_path, folder_lenth);
    cursor += folder_lenth;

    for (int i = 0; i < n_models; ++i) {
        const char *name = TextDatumGetCString(datum_model_names[i]);
        uint32 name_len = strlen(name) + 1;
        memcpy(cursor, &name_len, sizeof(uint32));
        cursor += sizeof(uint32);
        memcpy(cursor, name, name_len);
        cursor += name_len;
    }

    if (response_channel == NULL) {
        char resp_name[64];
        snprintf(resp_name, sizeof resp_name, "ns_resp_%u", MyProcPid);
        response_channel = NSChannelInit(resp_name, true);
    }

    NSTaskMsg *msg = NewTask(NS_TASK_SAVE_MODELS, payload_size, MyProcPid, payload);

    NSChannel *req_channel = NSChannelInit(NS_CHANNEL, false);
    if (!NSChannelPush(req_channel, msg))
        ereport(ERROR, (errmsg("NeurStore: push NS_TASK_SAVE_MODELS failed")));
    pfree(msg);
    pfree(payload);

    NSTaskMsg *response;
    for (;;) {
        response = NSChannelPop(response_channel);
        if (response != NULL)
            break;
        CHECK_FOR_INTERRUPTS();
    }
    bool ok = (response->type == NS_TASK_OK);
    pfree(response);

    if (!ok)
        PG_RETURN_BOOL(false);

    Relation rel = pgext_open("model", RowExclusiveLock);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_models; ++i) {
        char *model_name = text_to_cstring(DatumGetTextP(datum_model_names[i]));
        Datum nextval_result = pgext_nextval("model_model_id_seq");
        Datum values[2];
        bool nulls_insert[2] = {false, false};
        values[0] = Int32GetDatum((int32) nextval_result);
        values[1] = CStringGetTextDatum(model_name);
        pgext_insert(rel, values, nulls_insert);
    }
    pgext_close(rel, RowExclusiveLock);
    PG_RETURN_BOOL(true);
}

Datum
ns_save_models_from_folder(PG_FUNCTION_ARGS) {
    text *folder_path_text = PG_GETARG_TEXT_P(0);
    float4 tolerance = PG_GETARG_FLOAT4(1);
    char *folder_path = text_to_cstring(folder_path_text);

    DIR *dir = opendir(folder_path);
    if (dir == NULL) {
        elog(ERROR, "Could not open folder: %s", folder_path);
        PG_RETURN_BOOL(false);
    }

    List *model_names_list = NIL;
    struct dirent *entry;

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR)
            continue;
        char *filename = entry->d_name;
        int filename_len = strlen(filename);
        // Check for ".onnx"
        if (filename_len > 5 && strcmp(filename + filename_len - 5, ".onnx") == 0) {
            int model_name_len = filename_len - 5;
            char *model_name = (char *) palloc(model_name_len + 1);
            memcpy(model_name, filename, model_name_len);
            model_name[model_name_len] = '\0';
            model_names_list = lappend(model_names_list, model_name);
        }
    }
    closedir(dir);

    if (model_names_list == NIL) {
        elog(INFO, "No .onnx models found in folder: %s", folder_path);
        PG_RETURN_BOOL(true);
    }

    int n_models = list_length(model_names_list);
    Datum *model_names_datums = (Datum *) palloc(n_models * sizeof(Datum));
    ListCell *lc;
    int i = 0;
    foreach(lc, model_names_list) {
        char *model_name = (char *) lfirst(lc);
        model_names_datums[i++] = CStringGetTextDatum(model_name);
    }

    ArrayType *model_names_array = construct_array(
        model_names_datums,
        n_models,
        TEXTOID,
        -1,
        false,
        'i'
    );

    /*
     * Call ns_save_models with the model_names_array
     */
    Datum result = DirectFunctionCall3(
        ns_save_models,
        PointerGetDatum(model_names_array),
        Float4GetDatum(tolerance),
        PointerGetDatum(folder_path_text)
    );
    pfree(model_names_datums);
    pfree(folder_path);
    list_free_deep(model_names_list);
    PG_RETURN_DATUM(result);
}

Datum
ns_save_model_dry_run(PG_FUNCTION_ARGS) {
    float4 tolerance = PG_GETARG_FLOAT4(0);
    char *model_path = text_to_cstring(PG_GETARG_TEXT_P(1));
    int32 load_mode = PG_GETARG_INT32(2);

    char *query = text_to_cstring(PG_GETARG_TEXT_P(3));
    char *pg_dsn = text_to_cstring(PG_GETARG_TEXT_P(4));
    ArrayType *input_cols = PG_GETARG_ARRAYTYPE_P(5);
    char *output_column = text_to_cstring(PG_GETARG_TEXT_P(6));
    int32 batch_size = PG_GETARG_INT32(7);
    char *tokenizer_path = text_to_cstring(PG_GETARG_TEXT_P(8));
    char *pad_token = text_to_cstring(PG_GETARG_TEXT_P(9));
    char *eos_token = text_to_cstring(PG_GETARG_TEXT_P(10));
    char *bos_token = text_to_cstring(PG_GETARG_TEXT_P(11));
    int32 max_input_len = PG_GETARG_INT32(12);
    int32 max_output_len = PG_GETARG_INT32(13);
    int32 task = PG_GETARG_INT32(14);
    bool use_gpu = PG_GETARG_BOOL(15);

    if (ARR_NDIM(input_cols) != 1) {
        ereport(ERROR, (errmsg("input_columns must be 1-D array")));
    }

    int n_cols = ArrayGetNItems(ARR_NDIM(input_cols), ARR_DIMS(input_cols));
    Datum *col_datums;
    bool *col_nulls;
    deconstruct_array(
        input_cols,
        TEXTOID,
        -1,
        false,
        'i',
        &col_datums,
        &col_nulls,
        &n_cols
    );
    for (int i = 0; i < n_cols; ++i) {
        if (col_nulls[i]) {
            ereport(ERROR, (errmsg("input_columns[%d] is NULL", i)));
        }
    }

    /* payload size */
    uint32 model_path_len = strlen(model_path) + 1;
    uint32 model_id_len = 1;
    uint32 query_len = strlen(query) + 1;
    uint32 pg_dsn_len = strlen(pg_dsn) + 1;
    uint32 output_column_len = strlen(output_column) + 1;
    uint32 tokenizer_path_len = strlen(tokenizer_path) + 1;
    uint32 pad_token_len = strlen(pad_token) + 1;
    uint32 eos_token_len = strlen(eos_token) + 1;
    uint32 bos_token_len = strlen(bos_token) + 1;

    Size payload_sz = sizeof(SaveModelDryRunPayload) +
                      model_path_len + model_id_len + query_len + pg_dsn_len +
                      output_column_len + tokenizer_path_len +
                      pad_token_len + eos_token_len + bos_token_len;

    for (int i = 0; i < n_cols; ++i) {
        const char *col = TextDatumGetCString(col_datums[i]);
        payload_sz += sizeof(uint32) + strlen(col) + 1;
    }

    SaveModelDryRunPayload *payload = palloc0(payload_sz);
    payload->tolerance = tolerance;
    payload->load_mode = (uint32) load_mode;
    payload->model_path_len = model_path_len;

    payload->batch_size = (uint32) batch_size;
    payload->max_input_len = (uint32) max_input_len;
    payload->max_output_len = (uint32) max_output_len;
    payload->task = (uint32) task;
    payload->use_gpu = use_gpu ? 1U : 0U;

    payload->model_id_len = model_id_len;
    payload->query_len = query_len;
    payload->pg_dsn_len = pg_dsn_len;
    payload->output_column_len = output_column_len;
    payload->tokenizer_path_len = tokenizer_path_len;
    payload->pad_token_len = pad_token_len;
    payload->eos_token_len = eos_token_len;
    payload->bos_token_len = bos_token_len;
    payload->n_input_columns = (uint32) n_cols;

    char *cursor = payload->data;
#define COPY_STR(s, len) do { memcpy(cursor, (s), (len)); cursor += (len); } while (0)
    COPY_STR(model_path,      model_path_len);
    COPY_STR("",              model_id_len);
    COPY_STR(query,           query_len);
    COPY_STR(pg_dsn,          pg_dsn_len);
    COPY_STR(output_column,   output_column_len);
    COPY_STR(tokenizer_path,  tokenizer_path_len);
    COPY_STR(pad_token,       pad_token_len);
    COPY_STR(eos_token,       eos_token_len);
    COPY_STR(bos_token,       bos_token_len);

    for (int i = 0; i < n_cols; ++i) {
        const char *col = TextDatumGetCString(col_datums[i]);
        uint32 len = strlen(col) + 1;
        memcpy(cursor, &len, sizeof(uint32)); cursor += sizeof(uint32);
        COPY_STR(col, len);
    }
#undef COPY_STR

    /* prepare the response channel */
    if (response_channel == NULL) {
        char resp_name[64];
        snprintf(resp_name, sizeof resp_name, "ns_resp_%u", MyProcPid);
        response_channel = NSChannelInit(resp_name, true);
    }

    NSChannel *req_chan = NSChannelInit(NS_CHANNEL, false);
    NSTaskMsg *msg = NewTask(
        NS_TASK_SAVE_MODEL_DRY_RUN,
        payload_sz,
        MyProcPid,
        payload
    );
    if (!NSChannelPush(req_chan, msg)) {
        ereport(ERROR, (errmsg("push NS_TASK_SAVE_MODEL_DRY_RUN failed")));
    }
    pfree(msg);
    pfree(payload);

    NSTaskMsg *resp;
    for (;;) {
        resp = NSChannelPop(response_channel);
        if (resp) break;
        CHECK_FOR_INTERRUPTS();
    }

    if (resp->type != NS_TASK_OK) {
        elog(ERROR, "Dry-run failed: %s", (const char *) resp->payload);
    }

    double ratio = 0.0, delta = 0.0;
    memcpy(&ratio, resp->payload, sizeof(double));
    memcpy(&delta, resp->payload + sizeof(double), sizeof(double));
    pfree(resp);

    Datum vals[2] = { Float8GetDatum(ratio), Float8GetDatum(delta) };
    ArrayType *out = construct_array(
        vals,
        2,
        FLOAT8OID,
        sizeof(float8),
        FLOAT8PASSBYVAL,
        'd'
    );
    PG_RETURN_ARRAYTYPE_P(out);
}

static bool lookup_model_name(int model_id, char **model_name_out) {
    ScanKeyData skey;
    ScanKeyInit(&skey,
        1,
        BTEqualStrategyNumber,
        F_INT4EQ,
        Int32GetDatum(model_id)
    );

    ModelSelectResult sel = {0};
    sel.found = false;

    Oid idx_oid = get_relname_relid(
        "model_pkey",
        get_namespace_oid("public", false)
    );
    if (!OidIsValid(idx_oid))
        ereport(ERROR, (errmsg("Index model_pkey not found")));

    Relation rel = pgext_open("model", AccessShareLock);
    pgext_select(rel, idx_oid, &skey, 1, model_select_callback, &sel);
    pgext_close(rel, AccessShareLock);
    if (!sel.found)
        return false;
    *model_name_out = sel.model_name;
    return true;
}

static bytea *send_load_model_task(
    const char *model_name,
    uint32 load_mode,
    bool return_serialized
) {
    if (response_channel == NULL) {
        char resp_name[64];
        snprintf(resp_name, sizeof resp_name, "ns_resp_%u", MyProcPid);
        response_channel = NSChannelInit(resp_name, true);
    }

    uint32 name_len = strlen(model_name) + 1;
    Size   payload_szie  = sizeof(LoadModelPayload) + name_len;

    LoadModelPayload *payload = palloc0(payload_szie);
    payload->return_serialized = return_serialized ? 1 : 0;
    payload->load_mode         = load_mode;
    payload->name_len          = name_len;
    memcpy(payload->data, model_name, name_len);

    NSTaskMsg *msg = NewTask(
    NS_TASK_LOAD_MODEL,
        payload_szie,
        MyProcPid,
        payload
    );

    NSChannel *req_chan = NSChannelInit(NS_CHANNEL, false);
    if (!NSChannelPush(req_chan, msg))
        ereport(ERROR, (errmsg("push NS_TASK_LOAD_MODEL failed")));
    pfree(msg);
    pfree(payload);

    NSTaskMsg *resp;
    for (;;) {
        resp = NSChannelPop(response_channel);
        if (resp) break;
        CHECK_FOR_INTERRUPTS();
    }

    bytea *result_bytea = NULL;

    if (resp->type == NS_TASK_OK && return_serialized) {
        uint32 data_size;
        memcpy(&data_size, resp->payload, sizeof(uint32));

        result_bytea = (bytea *) palloc(VARHDRSZ + data_size);
        SET_VARSIZE(result_bytea, VARHDRSZ + data_size);
        memcpy(
            VARDATA(result_bytea),
            resp->payload + sizeof(uint32),
            data_size
        );
    } else if (resp->type != NS_TASK_OK)
    {
        elog(
            ERROR, "NeurStore: Task failed with error: %s",
            (const char *) resp->payload
        );
    }

    pfree(resp);
    return result_bytea;    // NULL if not return_serialized
}


/**
 * Load a model from the database
 * @param model_id int The id of the model
 * @return bytea The decompressed serialized model (in .ONNX format)
 */
Datum
ns_load_model(PG_FUNCTION_ARGS) {
    int model_id = PG_GETARG_INT32(0);
    bool return_serialized = PG_GETARG_BOOL(1);

    char *model_name;
    if (!lookup_model_name(model_id, &model_name))
        ereport(ERROR, (errmsg("Model with ID %d not found", model_id)));

    bytea *result = send_load_model_task(
        model_name,
        NS_LOAD_FLOAT32,
        return_serialized
    );
    pfree(model_name);
    if (!return_serialized)
        PG_RETURN_NULL();
    else
        PG_RETURN_BYTEA_P(result);
}

Datum
ns_load_model_as_uint8(PG_FUNCTION_ARGS) {
    int model_id = PG_GETARG_INT32(0);
    bool return_serialized = PG_GETARG_BOOL(1);

    char *model_name;
    if (!lookup_model_name(model_id, &model_name))
        ereport(ERROR, (errmsg("Model with ID %d not found", model_id)));

    bytea *result = send_load_model_task(
        model_name,
        NS_LOAD_UINT8,
        return_serialized
    );
    pfree(model_name);
    if (!return_serialized)
        PG_RETURN_NULL();
    else
        PG_RETURN_BYTEA_P(result);
}

Datum
ns_load_model_as_uint8_delta(PG_FUNCTION_ARGS) {
    int model_id = PG_GETARG_INT32(0);
    bool return_serialized = PG_GETARG_BOOL(1);

    char *model_name;
    if (!lookup_model_name(model_id, &model_name))
        ereport(ERROR, (errmsg("Model with ID %d not found", model_id)));

    bytea *result = send_load_model_task(
        model_name,
        NS_LOAD_UINT8_DELTA,
        return_serialized
    );
    pfree(model_name);
    if (!return_serialized)
        PG_RETURN_NULL();
    else
        PG_RETURN_BYTEA_P(result);
}

Datum
ns_load_model_as_float16(PG_FUNCTION_ARGS) {
    int model_id = PG_GETARG_INT32(0);
    bool return_serialized = PG_GETARG_BOOL(1);

    char *model_name;
    if (!lookup_model_name(model_id, &model_name))
        ereport(ERROR, (errmsg("Model with ID %d not found", model_id)));

    bytea *result = send_load_model_task(
        model_name,
        NS_LOAD_FLOAT16,
        return_serialized
    );
    pfree(model_name);
    if (!return_serialized)
        PG_RETURN_NULL();
    else
        PG_RETURN_BYTEA_P(result);
}

Datum
ns_inference(PG_FUNCTION_ARGS) {
    int   model_id_int = PG_GETARG_INT32(0);
    char *model_id;
    if (!lookup_model_name(model_id_int, &model_id)) {
        ereport(ERROR, (errmsg("Model with ID %d not found", model_id_int)));
    }

    char  *query          = text_to_cstring(PG_GETARG_TEXT_P(1));
    char  *pg_dsn         = text_to_cstring(PG_GETARG_TEXT_P(2));
    ArrayType *input_cols = PG_GETARG_ARRAYTYPE_P(3);
    char  *output_column  = text_to_cstring(PG_GETARG_TEXT_P(4));
    int32  batch_size     = PG_GETARG_INT32(5);
    char  *tokenizer_path = text_to_cstring(PG_GETARG_TEXT_P(6));
    char  *pad_token      = text_to_cstring(PG_GETARG_TEXT_P(7));
    char  *eos_token      = text_to_cstring(PG_GETARG_TEXT_P(8));
    char  *bos_token      = text_to_cstring(PG_GETARG_TEXT_P(9));
    int32  max_input_len  = PG_GETARG_INT32(10);
    int32  max_output_len = PG_GETARG_INT32(11);
    int32  load_mode      = PG_GETARG_INT32(12);
    int32  task           = PG_GETARG_INT32(13);
    bool   use_gpu        = PG_GETARG_BOOL(14);

    if (ARR_NDIM(input_cols) != 1) {
        ereport(ERROR, errmsg("input_columns must be 1-D array"));
    }

    int n_cols = ArrayGetNItems(ARR_NDIM(input_cols), ARR_DIMS(input_cols));
    Datum *col_datums;
    bool  *col_nulls;
    deconstruct_array(
        input_cols,
        TEXTOID,
        -1,
        false,
        'i',
        &col_datums,
        &col_nulls,
        &n_cols
    );

    for (int i = 0; i < n_cols; ++i) {
        if (col_nulls[i]) {
            ereport(ERROR, (errmsg("input_columns[%d] is NULL", i)));
        }
    }

    /* payload size */
    uint32 model_id_len       = strlen(model_id) + 1;
    uint32 query_len          = strlen(query) + 1;
    uint32 pg_dsn_len         = strlen(pg_dsn) + 1;
    uint32 output_column_len  = strlen(output_column) + 1;
    uint32 tokenizer_path_len = strlen(tokenizer_path) + 1;
    uint32 pad_token_len      = strlen(pad_token) + 1;
    uint32 eos_token_len      = strlen(eos_token) + 1;
    uint32 bos_token_len      = strlen(bos_token) + 1;

    Size payload_size = sizeof(InferencePayload) +
                   model_id_len + query_len + pg_dsn_len +
                   output_column_len + tokenizer_path_len +
                   pad_token_len + eos_token_len + bos_token_len;

    for (int i = 0; i < n_cols; ++i)
        payload_size += sizeof(uint32) +
                   strlen(TextDatumGetCString(col_datums[i])) + 1;

    /* build up payload */
    InferencePayload *payload = palloc0(payload_size);
    payload->batch_size      = batch_size;
    payload->max_input_len   = max_input_len;
    payload->max_output_len  = max_output_len;
    payload->load_mode       = load_mode;
    payload->task            = task;
    payload->use_gpu         = use_gpu ? 1 : 0;

    payload->model_id_len       = model_id_len;
    payload->query_len          = query_len;
    payload->pg_dsn_len         = pg_dsn_len;
    payload->output_column_len  = output_column_len;
    payload->tokenizer_path_len = tokenizer_path_len;
    payload->pad_token_len      = pad_token_len;
    payload->eos_token_len      = eos_token_len;
    payload->bos_token_len      = bos_token_len;
    payload->n_input_columns    = n_cols;

    char *cursor = payload->data;
#define COPY_STR(str,len) memcpy(cursor,(str),(len)); cursor += (len)
    COPY_STR(model_id,       model_id_len);
    COPY_STR(query,          query_len);
    COPY_STR(pg_dsn,         pg_dsn_len);
    COPY_STR(output_column,  output_column_len);
    COPY_STR(tokenizer_path, tokenizer_path_len);
    COPY_STR(pad_token,      pad_token_len);
    COPY_STR(eos_token,      eos_token_len);
    COPY_STR(bos_token,      bos_token_len);

    for (int i = 0; i < n_cols; ++i) {
        const char *col = TextDatumGetCString(col_datums[i]);
        uint32 len = strlen(col) + 1;
        memcpy(cursor, &len, sizeof(uint32));
        cursor += sizeof(uint32);
        COPY_STR(col, len);
    }
#undef COPY_STR

    /* prepare the response channel */
    if (response_channel == NULL) {
        char resp_name[64];
        snprintf(resp_name, sizeof resp_name, "ns_resp_%u", MyProcPid);
        response_channel = NSChannelInit(resp_name, true);
    }

    NSTaskMsg *msg = NewTask(
        NS_TASK_INFERNECE,
        payload_size,
        MyProcPid,
        payload
    );

    NSChannel *req_chan = NSChannelInit(NS_CHANNEL, false);
    if (!NSChannelPush(req_chan, msg)) {
        ereport(ERROR, errmsg("push NS_TASK_INFERNECE failed"));
    }

    pfree(msg);
    pfree(payload);

    NSTaskMsg *resp;
    for (;;) {
        resp = NSChannelPop(response_channel);
        if (resp) break;
        CHECK_FOR_INTERRUPTS();
    }
    bool ok = (resp->type == NS_TASK_OK);
    if (!ok) {
        elog(ERROR, "NeurStore inference failed: %s", (const char *) resp->payload);
    }
    pfree(resp);
    PG_RETURN_BOOL(ok);
}

Datum
ns_clean_cache(PG_FUNCTION_ARGS) {
    if (response_channel == NULL) {
        char resp_name[64];
        snprintf(resp_name, sizeof resp_name, "ns_resp_%u", MyProcPid);
        response_channel = NSChannelInit(resp_name, true);
    }

    NSTaskMsg *msg = NewTask(NS_TASK_CLEAN_CACHE, 0, MyProcPid, NULL);

    NSChannel *req_chan = NSChannelInit(NS_CHANNEL, false);
    if (!NSChannelPush(req_chan, msg)) {
        ereport(ERROR, (errmsg("NeurStore: push clean cache task failed")));
    }
    pfree(msg);

    NSTaskMsg *response;
    for (;;) {
        response = NSChannelPop(response_channel);
        if (response != NULL)
            break;
        CHECK_FOR_INTERRUPTS();
    }

    bool ok = (response->type == NS_TASK_OK);
    pfree(response);

    PG_RETURN_BOOL(ok);
}


// Datum
// ns_set_parallelism(PG_FUNCTION_ARGS) {
//     int32 new_parallelism = PG_GETARG_INT32(0);
//     if (new_parallelism <= 0) {
//         ereport(ERROR, (errmsg("parallelism must be positive")));
//     }
//     bool success = set_parallelism(new_parallelism);
//     PG_RETURN_BOOL(success);
// }
//
// Datum
// ns_get_parallelism(PG_FUNCTION_ARGS) {
//     int current_parallelism = get_parallelism();
//     PG_RETURN_INT32(current_parallelism);
// }
