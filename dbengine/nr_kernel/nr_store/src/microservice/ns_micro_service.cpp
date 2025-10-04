#include "microservice/ns_micro_service.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <omp.h>

#include "microservice/ns_micro_proto.h"
#include "microservice/ns_micro_threadpool.h"
#include "neurstore/neurstore.h"
#include "neurstore/inference/inference_utils.h"


static IndexCacheManagerC *global_cache_mgr = nullptr;
static char *global_store_path = nullptr;
static MicroThreadPool global_thread_pool;
static int global_service_inited = 0;
static int global_omp_parallelism = 1;

typedef struct ServiceTask {
    NSMicroTaskMsg *task_message;
    ns_micro_response_callback callback;
    void *context;
} ServiceTask;

static NSMicroTaskMsg *handle_save_model(const NSMicroTaskMsg *task) {
    if (task->header.payload_size < sizeof(SaveModelPayload)) {
        return NewMicroErrorResponse("invalid payload size: too small");
    }

    auto payload = reinterpret_cast<const SaveModelPayload *>(task->payload);
    const size_t expect = sizeof(SaveModelPayload) + payload->name_len + payload->path_len;

    if (task->header.payload_size < expect || payload->name_len == 0 || payload->path_len == 0) {
        return NewMicroErrorResponse("invalid name/path length");
    }

    double tolerance = payload->tolerance;
    const char *name_ptr = payload->data;
    const char *path_ptr = payload->data + payload->name_len;

    NeurStoreC *ns = ns_create(global_store_path, global_cache_mgr);
    bool ok = ns_save_model_internal(ns, name_ptr, tolerance, path_ptr);
    ns_destroy(ns);

    return ok ? NewMicroOkResponse() : NewMicroErrorResponse("ns_save_model_internal failed");
}

static NSMicroTaskMsg *handle_save_models(const NSMicroTaskMsg *task) {
    if (task->header.payload_size < sizeof(SaveModelsPayload)) {
        return NewMicroErrorResponse("invalid payload size: too small");
    }

    auto payload = reinterpret_cast<const SaveModelsPayload *>(task->payload);
    const char *cursor = payload->data;

    if (payload->folder_len == 0 || payload->n_models == 0) {
        return NewMicroErrorResponse("invalid folder_len or n_models");
    }

    if (task->header.payload_size < sizeof(SaveModelsPayload) + payload->folder_len + payload->n_models * sizeof(
            uint32_t)) {
        return NewMicroErrorResponse("payload truncated");
    }

    const char *folder_path = cursor;
    cursor += payload->folder_len;

    auto model_names = static_cast<char **>(malloc(payload->n_models * sizeof(char *)));

    for (uint32_t i = 0; i < payload->n_models; ++i) {
        uint32_t name_len = 0;
        if (cursor + sizeof(uint32_t) > reinterpret_cast<const char *>(task->payload) + task->header.payload_size) {
            free(model_names);
            return NewMicroErrorResponse("payload truncated");
        }
        memcpy(&name_len, cursor, sizeof(uint32_t));
        cursor += sizeof(uint32_t);

        if (name_len == 0 || cursor + name_len > reinterpret_cast<const char *>(task->payload) + task->header.
            payload_size) {
            free(model_names);
            return NewMicroErrorResponse("invalid name_len or payload overflow");
        }
        model_names[i] = const_cast<char *>(cursor);
        cursor += name_len;
    }

    NeurStoreC *ns = ns_create(global_store_path, global_cache_mgr);
    bool ok = ns_save_models_internal(
        ns,
        const_cast<const char **>(model_names),
        static_cast<int>(payload->n_models),
        payload->tolerance,
        folder_path
    );
    ns_destroy(ns);
    free(model_names);
    return ok ? NewMicroOkResponse() : NewMicroErrorResponse("ns_save_models_internal failed");
}

static NSMicroTaskMsg *handle_load_model(const NSMicroTaskMsg *task) {
    if (task->header.payload_size < sizeof(LoadModelPayload)) {
        return NewMicroErrorResponse("invalid payload size: too small");
    }

    auto payload = reinterpret_cast<const LoadModelPayload *>(task->payload);
    if (payload->name_len == 0 || payload->load_mode > NS_LOAD_FLOAT16) {
        return NewMicroErrorResponse("invalid field");
    }

    if (task->header.payload_size < sizeof(LoadModelPayload) + payload->name_len) {
        return NewMicroErrorResponse("payload truncated");
    }

    const char *model_name = payload->data;

    NeurStoreC *ns = ns_create(global_store_path, global_cache_mgr);
    ModelC *model = nullptr;

    switch (payload->load_mode) {
        case NS_LOAD_FLOAT32: model = ns_load_model_internal(ns, model_name);
            break;
        case NS_LOAD_UINT8: model = ns_load_model_intermal_uint8(ns, model_name);
            break;
        case NS_LOAD_UINT8_DELTA: model = ns_load_model_intermal_uint8_delta(ns, model_name);
            break;
        case NS_LOAD_FLOAT16: model = ns_load_model_intermal_float16(ns, model_name);
            break;
        default: break;
    }

    if (!model) {
        ns_destroy(ns);
        return NewMicroErrorResponse("ns_load_model failed");
    }

    if (payload->return_serialized == 0) {
        m_destroy_model(model);
        ns_destroy(ns);
        return NewMicroOkResponse();
    } else {
        size_t out_size = 0;
        char *blob = m_serialize(model, &out_size);

        const uint32_t resp_payload_sz = static_cast<uint32_t>(sizeof(uint32_t) + out_size);
        NSMicroTaskMsg *resp = NewMicroTask(NS_MICRO_TASK_OK, resp_payload_sz, 0, nullptr);
        memcpy(resp->payload, &out_size, sizeof(uint32_t));
        memcpy(resp->payload + sizeof(uint32_t), blob, out_size);
        free(blob);
        m_destroy_model(model);
        ns_destroy(ns);
        return resp;
    }
}

static NSMicroTaskMsg *handle_inference(const NSMicroTaskMsg *task) {
    if (task->header.payload_size < sizeof(InferencePayload))
        return NewMicroErrorResponse("payload too small");

    auto payload = reinterpret_cast<const InferencePayload *>(task->payload);
    const char *cursor = payload->data;

#define ADVANCE_STR(dest, len) do { \
    (dest) = (char*)cursor; \
    cursor += (len); \
    if (cursor > ((const char*)task->payload + task->header.payload_size)) \
        return NewMicroErrorResponse("payload truncated"); \
} while (0)

    char *model_id, *query, *pg_dsn, *output_column, *tokenizer_path, *pad_token, *eos_token, *bos_token;

    ADVANCE_STR(model_id, payload->model_id_len);
    ADVANCE_STR(query, payload->query_len);
    ADVANCE_STR(pg_dsn, payload->pg_dsn_len);
    ADVANCE_STR(output_column, payload->output_column_len);
    ADVANCE_STR(tokenizer_path, payload->tokenizer_path_len);
    ADVANCE_STR(pad_token, payload->pad_token_len);
    ADVANCE_STR(eos_token, payload->eos_token_len);
    ADVANCE_STR(bos_token, payload->bos_token_len);

    char **input_columns = nullptr;
    if (payload->n_input_columns > 0) {
        input_columns = static_cast<char **>(malloc(payload->n_input_columns * sizeof(char *)));
    }
    for (uint32_t i = 0; i < payload->n_input_columns; ++i) {
        uint32_t col_len;
        if (cursor + sizeof(uint32_t) > reinterpret_cast<const char *>(task->payload) + task->header.payload_size) {
            free(input_columns);
            return NewMicroErrorResponse("payload truncated");
        }
        memcpy(&col_len, cursor, sizeof(uint32_t));
        cursor += sizeof(uint32_t);
        if (cursor + col_len > reinterpret_cast<const char *>(task->payload) + task->header.payload_size)
            return NewMicroErrorResponse("payload truncated");
        input_columns[i] = const_cast<char *>(cursor);
        cursor += col_len;
    }
#undef ADVANCE_STR

    InferenceOptionsC *opts = ioc_create(
        model_id,
        query,
        pg_dsn,
        const_cast<const char **>(input_columns), static_cast<int>(payload->n_input_columns),
        output_column,
        static_cast<int>(payload->batch_size),
        tokenizer_path,
        pad_token, eos_token, bos_token,
        static_cast<int>(payload->max_input_len),
        static_cast<int>(payload->max_output_len),
        static_cast<int>(payload->load_mode),
        static_cast<int>(payload->task),
        static_cast<int>(payload->use_gpu)
    );

    NeurStoreC *ns = ns_create(global_store_path, global_cache_mgr);
    const char *err_msg = nullptr;
    int ok = ns_inference_internal(ns, opts, &err_msg);
    ns_destroy(ns);
    ioc_destroy(opts);
    if (input_columns) {
        free(input_columns);
    }
    if (!ok) {
        NSMicroTaskMsg *err = NewMicroErrorResponse(err_msg ? err_msg : "ns_inference_internal failed");
        if (err_msg) {
            free((void *)err_msg);
        }
        return err;
    }
    return NewMicroOkResponse();
}

static NSMicroTaskMsg *handle_clean_cache() {
    if (global_cache_mgr) {
        icm_clear_cache(global_cache_mgr);
        return NewMicroOkResponse();
    } else {
        return NewMicroErrorResponse("Cache manager not initialized");
    }
}

static NSMicroTaskMsg *process_request(const NSMicroTaskMsg *req) {
    switch (req->header.type) {
        case NS_MICRO_TASK_SAVE_MODEL: return handle_save_model(req);
        case NS_MICRO_TASK_SAVE_MODELS: return handle_save_models(req);
        case NS_MICRO_TASK_LOAD_MODEL: return handle_load_model(req);
        case NS_MICRO_TASK_INFERENCE: return handle_inference(req);
        case NS_MICRO_TASK_CLEAN_CACHE: return handle_clean_cache();
        case NS_MICRO_TASK_SHUTDOWN: return NewMicroOkResponse();
        default: return NewMicroErrorResponse("unknown task type");
    }
}

static void *service_worker(void *arg) {
    auto task = static_cast<ServiceTask *>(arg);
    if (!task) {
        return nullptr;
    }

    NSMicroTaskMsg *task_msg = process_request(task->task_message);
    if (task->callback) {
        task->callback(task_msg, task->context);
    }
    if (task_msg) {
        ns_micro_msg_free(task_msg);
    }
    if (task->task_message) {
        ns_micro_msg_free(task->task_message);
    }
    free(task);
    return nullptr;
}

int ns_micro_service_init(const char *store_path, int num_threads, int omp_parallelism) {
    if (global_service_inited) {
        return 0;
    }
    if (!store_path || num_threads <= 0 || omp_parallelism <= 0) {
        return -1;
    }

    global_store_path = static_cast<char *>(malloc(strlen(store_path) + 1));
    memcpy(global_store_path, store_path, strlen(store_path) + 1);

    global_cache_mgr = icm_create(global_store_path);
    global_omp_parallelism = omp_parallelism;
    micro_threadpool_init(&global_thread_pool, num_threads, global_omp_parallelism);
    global_service_inited = 1;
    return 0;
}

void ns_micro_service_shutdown() {
    if (!global_service_inited) return;
    micro_threadpool_destroy(&global_thread_pool);
    if (global_cache_mgr) {
        icm_destroy(global_cache_mgr);
        global_cache_mgr = nullptr;
    }
    if (global_store_path) {
        free(global_store_path);
        global_store_path = nullptr;
    }
    global_service_inited = 0;
}

int ns_micro_service_submit(NSMicroTaskMsg *task_msg, ns_micro_response_callback callback, void *context) {
    if (!global_service_inited || !task_msg) return -1;
    auto task = static_cast<ServiceTask *>(malloc(sizeof(ServiceTask)));
    task->task_message = task_msg;
    task->callback = callback;
    task->context = context;
    micro_threadpool_add_task(&global_thread_pool, service_worker, task);
    return 0;
}

void ns_micro_service_set_omp_parallelism(int omp_parallelism) {
    if (omp_parallelism > 0) {
        global_omp_parallelism = omp_parallelism;
        global_thread_pool.omp_parallelism.store(omp_parallelism);
    }
}
