#ifndef SERVICE_H
#define SERVICE_H

#include <postgres.h>


void run_neurstore(int num_threads, int omp_parallelism, const char *store_path);

void *process_task(void *arg);

void stop_neurstore(void);

/**
 * [double tolerance, uint32 name_len, uint32 path_len, [name, path]]
 */
typedef struct SaveModelPayload {
    double tolerance;
    uint32 name_len;
    uint32 path_len;
    char data[FLEXIBLE_ARRAY_MEMBER];
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
    uint32 folder_len;
    uint32 n_models;
    char data[FLEXIBLE_ARRAY_MEMBER];
} SaveModelsPayload;

/**
 * LoadModelPayload layout
 *   uint32 return_serialized; (either 0 or 1)
 *   uint32 load_mode;
 *   uint32 name_len;
 *   char   model_name[name_len];
 */
typedef struct LoadModelPayload {
    uint32 return_serialized;
    uint32 load_mode;
    uint32 name_len;
    char   data[FLEXIBLE_ARRAY_MEMBER];
} LoadModelPayload;

typedef enum NSLoadMode {
    NS_LOAD_FLOAT32      = 0,
    NS_LOAD_UINT8        = 1,
    NS_LOAD_UINT8_DELTA  = 2,
    NS_LOAD_FLOAT16      = 3
} NSLoadMode;

typedef struct InferencePayload {
    uint32 batch_size;
    uint32 max_input_len;
    uint32 max_output_len;
    uint32 load_mode;
    uint32 task;
    uint32 use_gpu;

    uint32 model_id_len;
    uint32 query_len;
    uint32 pg_dsn_len;
    uint32 output_column_len;
    uint32 tokenizer_path_len;
    uint32 pad_token_len;
    uint32 eos_token_len;
    uint32 bos_token_len;
    uint32 n_input_columns;

    char   data[FLEXIBLE_ARRAY_MEMBER];
} InferencePayload;

typedef struct SaveModelDryRunPayload {
    float  tolerance;
    uint32 load_mode;
    uint32 model_path_len;

    uint32 batch_size;
    uint32 max_input_len;
    uint32 max_output_len;
    uint32 task;
    uint32 use_gpu;

    uint32 model_id_len;
    uint32 query_len;
    uint32 pg_dsn_len;
    uint32 output_column_len;
    uint32 tokenizer_path_len;
    uint32 pad_token_len;
    uint32 eos_token_len;
    uint32 bos_token_len;
    uint32 n_input_columns;

    char   data[FLEXIBLE_ARRAY_MEMBER];
} SaveModelDryRunPayload;

#endif //SERVICE_H
