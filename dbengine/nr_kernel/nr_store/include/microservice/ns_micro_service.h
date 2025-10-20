#ifndef NS_SERVICE_H
#define NS_SERVICE_H

#include <microservice/ns_micro_proto.h>


int ns_micro_service_init(const char *store_path, int num_threads, int omp_parallelism);

void ns_micro_service_shutdown();

typedef void (*ns_micro_response_callback)(NSMicroTaskMsg *response, void *context);

int ns_micro_service_submit(NSMicroTaskMsg *task_msg, ns_micro_response_callback callback, void *context);

void ns_micro_service_set_omp_parallelism(int omp_parallelism);

typedef enum NSLoadMode {
    NS_LOAD_FLOAT32 = 0,
    NS_LOAD_UINT8 = 1,
    NS_LOAD_UINT8_DELTA = 2,
    NS_LOAD_FLOAT16 = 3
} NSLoadMode;

/**
 * [double tolerance, uint32 name_len, uint32 path_len, [name, path]]
 */
typedef struct SaveModelPayload {
    double tolerance;
    uint32_t name_len;
    uint32_t path_len;
    char data[];
} SaveModelPayload;

/**
 * [double  tolerance, uint32  folder_len, uint32  n_models,
 *   [char folder_path[folder_len];
 *    REPEAT n_models TIMES {
 *      uint32 name_len;
 *      char name[name_len];
 *    }
 *   ]
 * ]
 */
typedef struct SaveModelsPayload {
    double tolerance;
    uint32_t folder_len;
    uint32_t n_models;
    char data[];
} SaveModelsPayload;

/**
 * LoadModelPayload layout
 *   uint32 return_serialized; (either 0 or 1)
 *   uint32 load_mode;
 *   uint32 name_len;
 *   char   model_name[name_len];
 */
typedef struct LoadModelPayload {
    uint32_t return_serialized;
    uint32_t load_mode;
    uint32_t name_len;
    char data[];
} LoadModelPayload;

typedef struct InferencePayload {
    uint32_t batch_size;
    uint32_t max_input_len;
    uint32_t max_output_len;
    uint32_t load_mode;
    uint32_t task;
    uint32_t use_gpu;

    uint32_t model_id_len;
    uint32_t query_len;
    uint32_t pg_dsn_len;
    uint32_t output_column_len;
    uint32_t tokenizer_path_len;
    uint32_t pad_token_len;
    uint32_t eos_token_len;
    uint32_t bos_token_len;
    uint32_t n_input_columns;

    char data[];
} InferencePayload;

#endif //NS_SERVICE_H
