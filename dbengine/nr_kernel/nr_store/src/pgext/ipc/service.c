#include "pgext/ipc/service.h"

#include <miscadmin.h>
#include <storage/ipc.h>
#include <utils/memutils.h>

#include "neurstore/neurstore.h"
#include "neurstore/inference/inference_utils.h"
#include "pgext/ipc/channel.h"
#include "pgext/ipc/threadpool.h"
#include "pgext/global.h"


static IndexCacheManagerC *global_cache_mgr = NULL;
static char *global_store_path = NULL;
static volatile sig_atomic_t ns_service_running = false;

typedef struct WorkerContext {
    NSTaskMsg *task;
    NSChannel *resp_channel;
} WorkerContext;

void run_neurstore(int num_threads, int omp_parallelism, const char *store_path) {
    global_store_path = MemoryContextStrdup(TopMemoryContext, store_path);
    global_cache_mgr = icm_create(global_store_path);
    ThreadPool pool;
    NSChannel *channel = NSChannelInit(NS_CHANNEL, false);

    threadpool_init(&pool, num_threads, omp_parallelism);
    ns_service_running = true;
    while (ns_service_running) {
        NSTaskMsg *task = NSChannelPop(channel);
        if (task == NULL) {
            CHECK_FOR_INTERRUPTS();
            continue;
        }
        if (task->type == NS_TASK_SHUTDOWN) {
            ns_service_running = false;
            pfree(task);
            break;
        }

        // Init response channel:
        // The response channel must be initialized before being handed over to the worker thread,
        // as the worker thread is not registered in the PostgreSQL background worker list.
        // To be more robust, we can't send the response back in the worker thread neither,
        // this is left as TODO.
        char resp_name[64];
        snprintf(resp_name, sizeof(resp_name), "ns_resp_%u", task->resp_channel);
        NSChannel *resp_channel = NSChannelInit(resp_name, false);
        WorkerContext *context = (WorkerContext *) palloc0(sizeof(WorkerContext));
        context->task = task;
        context->resp_channel = resp_channel;
        threadpool_add_task( &pool, process_task, context);
        CHECK_FOR_INTERRUPTS();
    }
    threadpool_destroy(&pool);
    NSChannelDestroy(channel);
    proc_exit(0);
}

static NSTaskMsg *handle_save_model(const NSTaskMsg *task) {
    if (task->payload_size < sizeof(SaveModelPayload))
        return NewErrorResponse("payload too small");

    const SaveModelPayload *payload = (const SaveModelPayload *) task->payload;

    const size_t expect = sizeof(SaveModelPayload) + payload->name_len + payload->path_len;
    if (task->payload_size < expect || payload->name_len == 0 || payload->path_len == 0) {
        return NewErrorResponse("invalid name/path length");
    }

    float tolerance = payload->tolerance;
    const char *name_ptr = payload->data;
    const char *path_ptr = payload->data + payload->name_len;

    NeurStoreC *ns = ns_create(global_store_path, global_cache_mgr);
    bool ok = ns_save_model_internal(ns, name_ptr, tolerance, path_ptr);
    ns_destroy(ns);

    if (!ok)
        return NewErrorResponse("save_model_internal failed");

    return NewOkResponse();
}

static NSTaskMsg *handle_save_models(const NSTaskMsg *task) {
    if (task->payload_size < sizeof(SaveModelsPayload))
        return NewErrorResponse("payload too small");

    const SaveModelsPayload *payload = (const SaveModelsPayload *) task->payload;
    const char *cursor = payload->data;

    if (payload->folder_len == 0 || payload->n_models == 0)
        return NewErrorResponse("invalid folder_len or n_models");

    if (task->payload_size < sizeof(SaveModelsPayload) + payload->folder_len + payload->n_models * sizeof(uint32))
        return NewErrorResponse("payload truncated");

    const char *folder_path = cursor;
    cursor += payload->folder_len;

    char **model_names = (char **) palloc(payload->n_models * sizeof(char *));
    for (uint32 i = 0; i < payload->n_models; ++i) {
        uint32 name_len;
        memcpy(&name_len, cursor, sizeof(uint32));
        cursor += sizeof(uint32);

        if (name_len == 0 || (cursor + name_len) > ((const char *) payload + task->payload_size)) {
            pfree(model_names);
            return NewErrorResponse("invalid name_len or payload overflow");
        }
        model_names[i] = (char *) cursor;
        cursor += name_len;
    }

    NeurStoreC *ns = ns_create(global_store_path, global_cache_mgr);
    bool ok = ns_save_models_internal(
        ns,
        (const char **) model_names,
        (int) payload->n_models,
        payload->tolerance,
        folder_path
    );
    ns_destroy(ns);
    pfree(model_names);
    return ok ? NewOkResponse()
              : NewErrorResponse("ns_save_models_internal failed");
}

static NSTaskMsg *handle_load_model(const NSTaskMsg *task) {
    if (task->payload_size < sizeof(LoadModelPayload))
        return NewErrorResponse("payload too small");

    const LoadModelPayload *payload = (const LoadModelPayload *) task->payload;

    if (payload->name_len == 0 || payload->load_mode > NS_LOAD_FLOAT16)
        return NewErrorResponse("invalid field");

    if (task->payload_size < sizeof(LoadModelPayload) + payload->name_len)
        return NewErrorResponse("payload truncated");

    const char *model_name = payload->data;

    NeurStoreC *ns = ns_create(global_store_path, global_cache_mgr);
    ModelC *model = NULL;

    switch (payload->load_mode) {
        case NS_LOAD_FLOAT32:
            model = ns_load_model_internal(ns, model_name);
            break;
        case NS_LOAD_UINT8:
            model = ns_load_model_intermal_uint8(ns, model_name);
            break;
        case NS_LOAD_UINT8_DELTA:
            model = ns_load_model_intermal_uint8_delta(ns, model_name);
            break;
        case NS_LOAD_FLOAT16:
            model = ns_load_model_intermal_float16(ns, model_name);
            break;
        default:
            break;
    }

    if (!model) {
        ns_destroy(ns);
        return NewErrorResponse("ns_load_model_xxx failed");
    }

    if (payload->return_serialized == 0) {
        m_destroy_model(model);
        ns_destroy(ns);
        return NewOkResponse();
    } else {
        // return serialized model
        size_t out_size = 0;
        char *blob = m_serialize(model, &out_size);

        Size resp_payload_sz = sizeof(uint32) + out_size;
        NSTaskMsg *resp = NewTask(
            NS_TASK_OK,
            resp_payload_sz,
            0,
            NULL
        );
        memcpy(resp->payload, &out_size, sizeof(uint32));
        memcpy(resp->payload + sizeof(uint32), blob, out_size);
        free(blob);
        m_destroy_model(model);
        ns_destroy(ns);
        return resp;
    }
}

static NSTaskMsg *handle_inference(const NSTaskMsg *task) {
    if (task->payload_size < sizeof(InferencePayload))
        return NewErrorResponse("payload too small");

    const InferencePayload *payload = (const InferencePayload *) task->payload;
    const char *cursor = payload->data;

#define ADVANCE_STR(dest, len)             \
    do {                                   \
        (dest) = (char *) cursor;          \
        cursor += (len);                   \
        if (cursor > ((const char *)task->payload + task->payload_size)) \
            return NewErrorResponse("payload truncated");                \
    } while (0)

    char *model_id, *query, *pg_dsn, *output_column,
        *tokenizer_path, *pad_token, *eos_token, *bos_token;

    ADVANCE_STR(model_id, payload->model_id_len);
    ADVANCE_STR(query, payload->query_len);
    ADVANCE_STR(pg_dsn, payload->pg_dsn_len);
    ADVANCE_STR(output_column, payload->output_column_len);
    ADVANCE_STR(tokenizer_path, payload->tokenizer_path_len);
    ADVANCE_STR(pad_token, payload->pad_token_len);
    ADVANCE_STR(eos_token, payload->eos_token_len);
    ADVANCE_STR(bos_token, payload->bos_token_len);

    char **input_columns = NULL;
    if (payload->n_input_columns > 0) {
        input_columns = (char **) palloc(payload->n_input_columns * sizeof(char *));
    }

    for (uint32 i = 0; i < payload->n_input_columns; ++i) {
        uint32 col_len;
        if (cursor + sizeof(uint32) > (const char *) task->payload + task->payload_size)
            return NewErrorResponse("payload truncated");

        memcpy(&col_len, cursor, sizeof(uint32));
        cursor += sizeof(uint32);

        if (cursor + col_len > (const char *) task->payload + task->payload_size)
            return NewErrorResponse("payload truncated");

        input_columns[i] = (char *) cursor;
        cursor += col_len;
    }
#undef ADVANCE_STR

    InferenceOptionsC *opts = ioc_create(
        model_id,
        query,
        pg_dsn,
        (const char **) input_columns, (int) payload->n_input_columns,
        output_column,
        (int) payload->batch_size,
        tokenizer_path,
        pad_token, eos_token, bos_token,
        (int) payload->max_input_len,
        (int) payload->max_output_len,
        (int) payload->load_mode,
        (int) payload->task,
        (int) payload->use_gpu
    );

    NeurStoreC *ns = ns_create(global_store_path, global_cache_mgr);

    const char *err_msg = NULL;
    int ok = ns_inference_internal(ns, opts, &err_msg);

    ns_destroy(ns);
    ioc_destroy(opts);
    if (input_columns) pfree(input_columns);

    if (!ok) {
        return NewErrorResponse(err_msg ? err_msg : "ns_inference_run failed");
    }
    return NewOkResponse();
}

static NSTaskMsg *handle_save_model_dry_run(const NSTaskMsg *task) {
    if (task->payload_size < sizeof(SaveModelDryRunPayload))
        return NewErrorResponse("payload too small");

    const SaveModelDryRunPayload *payload = (const SaveModelDryRunPayload *) task->payload;
    const char *cursor = payload->data;

#define ADVANCE_STR(dest, len)                                                   \
    do {                                                                         \
        (dest) = (char *) cursor;                                                \
        cursor += (len);                                                         \
        if (cursor > ((const char *)task->payload + task->payload_size))         \
            return NewErrorResponse("payload truncated");                        \
    } while (0)

    char *model_path, *model_id, *query, *pg_dsn, *output_column,
         *tokenizer_path, *pad_token, *eos_token, *bos_token;

    ADVANCE_STR(model_path,     payload->model_path_len);
    ADVANCE_STR(model_id,       payload->model_id_len);
    ADVANCE_STR(query,          payload->query_len);
    ADVANCE_STR(pg_dsn,         payload->pg_dsn_len);
    ADVANCE_STR(output_column,  payload->output_column_len);
    ADVANCE_STR(tokenizer_path, payload->tokenizer_path_len);
    ADVANCE_STR(pad_token,      payload->pad_token_len);
    ADVANCE_STR(eos_token,      payload->eos_token_len);
    ADVANCE_STR(bos_token,      payload->bos_token_len);

    char **input_columns = NULL;
    if (payload->n_input_columns > 0) {
        input_columns = (char **) palloc(payload->n_input_columns * sizeof(char *));
    }

    for (uint32 i = 0; i < payload->n_input_columns; ++i) {
        uint32 col_len;
        if (cursor + sizeof(uint32) > (const char *) task->payload + task->payload_size)
            return NewErrorResponse("payload truncated");
        memcpy(&col_len, cursor, sizeof(uint32));
        cursor += sizeof(uint32);

        if (cursor + col_len > (const char *) task->payload + task->payload_size)
            return NewErrorResponse("payload truncated");
        input_columns[i] = (char *) cursor;
        cursor += col_len;
    }
#undef ADVANCE_STR

    InferenceOptionsC *opts = ioc_create(
        model_id,
        query,
        pg_dsn,
        (const char **) input_columns, (int) payload->n_input_columns,
        output_column,
        (int) payload->batch_size,
        tokenizer_path,
        pad_token, eos_token, bos_token,
        (int) payload->max_input_len,
        (int) payload->max_output_len,
        (int) payload->load_mode,
        (int) payload->task,
        (int) payload->use_gpu
    );

    NeurStoreC *ns = ns_create(global_store_path, global_cache_mgr);

    double ratio = 0.0, delta_perf = 0.0;
    bool ok = ns_save_model_dry_run_internal(
        ns,
        payload->tolerance,
        model_path,
        payload->load_mode,
        opts,
        &ratio,
        &delta_perf
    );

    ns_destroy(ns);
    ioc_destroy(opts);
    if (input_columns) pfree(input_columns);

    if (!ok) {
        return NewErrorResponse("ns_save_model_dry_run failed");
    }

    Size resp_sz = sizeof(double) * 2;
    NSTaskMsg *resp = NewTask(NS_TASK_OK, resp_sz, 0, NULL);
    memcpy(resp->payload, &ratio, sizeof(double));
    memcpy(resp->payload + sizeof(double), &delta_perf, sizeof(double));
    return resp;
}

void *process_task(void *arg) {
    WorkerContext *context = (WorkerContext *) arg;
    NSTaskMsg *task = context->task;
    NSTaskMsg *resp = NULL;
    NSChannel *resp_channel = context->resp_channel;

    switch (task->type) {
        case NS_TASK_SAVE_MODEL:
            resp = handle_save_model(task);
            break;
        case NS_TASK_SAVE_MODELS:
            resp = handle_save_models(task);
            break;
        case NS_TASK_SAVE_MODEL_DRY_RUN:
            resp = handle_save_model_dry_run(task);
            break;
        case NS_TASK_LOAD_MODEL:
            resp = handle_load_model(task);
            break;
        case NS_TASK_INFERNECE:
            resp = handle_inference(task);
            break;
        case NS_TASK_CLEAN_CACHE:
            if (global_cache_mgr) {
                icm_clear_cache(global_cache_mgr);
                resp = NewOkResponse();
            } else {
                resp = NewErrorResponse("Cache manager not initialized");
            }
            break;
        case NS_TASK_SHUTDOWN:
            break;
        default:
            break;
    }
    if (resp != NULL) {
        NSChannelPush(resp_channel, resp);
        pfree(resp);
    }
    pfree(task);
    pfree(context);
    return NULL;
}
