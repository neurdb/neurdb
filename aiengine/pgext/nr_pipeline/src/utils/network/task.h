#ifndef TASK_H
#define TASK_H

#include <cjson/cJSON.h>


const char *ML_TASK[] = {"train", "inference", "finetune"};

/**
 * The machine learning task
 * T_TRAIN: Train the model
 * T_INFERENCE: Inference with the model
 * T_FINETUNE: Finetune the model
 */
typedef enum {
    T_TRAIN = 0,
    T_INFERENCE = 1,
    T_FINETUNE = 2
} MLTask;

typedef struct {
    char *architecture;
    int batch_size;
    int epoch;
    int n_batch_train;
    int n_batch_eval;
    int n_batch_test;
    double learning_rate;
    char *optimizer;
    char *loss;
    char *metrics;
} TrainTaskSpec;

typedef struct {
    char *architecture;
    int batch_size;
    int n_batch;
    char *metrics;
} InferenceTaskSpec;

typedef struct {
    char *model_name;
    int model_id;
    int batch_size;
    int epochs;
    int n_batch_train;
    int n_batch_eval;
    int n_batch_test;
    double learning_rate;
    char *optimizer;
    char *loss;
    char *metrics;
} FinetuneTaskSpec;

TrainTaskSpec *create_train_task_spec(
    const char *architecture,
    int batch_size,
    int epoch,
    int n_batch_train,
    int n_batch_eval,
    int n_batch_test,
    double learning_rate,
    const char *optimizer,
    const char *loss,
    const char *metrics
);

InferenceTaskSpec *create_inference_task_spec(
    const char *architecture,
    int batch_size,
    int n_batch,
    const char *metrics
);

FinetuneTaskSpec *create_finetune_task_spec(
    const char *model_name,
    int model_id,
    int batch_size,
    int epochs,
    int n_batch_train,
    int n_batch_eval,
    int n_batch_test,
    double learning_rate,
    const char *optimizer,
    const char *loss,
    const char *metrics
);

/**
 * Append the task specification to the json object
 * @param json The json object to be appended
 * @param task_spec The task specification, it can be TrainTaskSpec, InferenceTaskSpec, or FinetuneTaskSpec
 * @param ml_task The machine learning task
 */
void task_append_to_json(cJSON *json, void *task_spec, MLTask ml_task);

#endif //TASK_H
