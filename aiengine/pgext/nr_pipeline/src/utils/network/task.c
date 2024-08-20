#include "task.h"

#include <stdlib.h>
#include <string.h>


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
) {
    TrainTaskSpec *task = (TrainTaskSpec *) malloc(sizeof(TrainTaskSpec));
    task->architecture = strdup(architecture);
    task->batch_size = batch_size;
    task->epoch = epoch;
    task->n_batch_train = n_batch_train;
    task->n_batch_eval = n_batch_eval;
    task->n_batch_test = n_batch_test;
    task->learning_rate = learning_rate;
    task->optimizer = strdup(optimizer);
    task->loss = strdup(loss);
    task->metrics = strdup(metrics);
    return task;
}

InferenceTaskSpec *create_inference_task_spec(
    const char *architecture,
    int batch_size,
    int n_batch,
    const char *metrics
) {
    InferenceTaskSpec *task = (InferenceTaskSpec *) malloc(sizeof(InferenceTaskSpec));
    task->architecture = strdup(architecture);
    task->batch_size = batch_size;
    task->n_batch = n_batch;
    task->metrics = strdup(metrics);
    return task;
}

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
) {
    FinetuneTaskSpec *task = (FinetuneTaskSpec *) malloc(sizeof(FinetuneTaskSpec));
    task->model_name = strdup(model_name);
    task->model_id = model_id;
    task->batch_size = batch_size;
    task->epochs = epochs;
    task->n_batch_train = n_batch_train;
    task->n_batch_eval = n_batch_eval;
    task->n_batch_test = n_batch_test;
    task->learning_rate = learning_rate;
    task->optimizer = strdup(optimizer);
    task->loss = strdup(loss);
    task->metrics = strdup(metrics);
    return task;
}

void free_train_task_spec(TrainTaskSpec *task) {
    free(task->architecture);
    free(task->optimizer);
    free(task->loss);
    free(task->metrics);
    free(task);
}

void free_inference_task_spec(InferenceTaskSpec *task) {
    free(task->architecture);
    free(task->metrics);
    free(task);
}

void free_finetune_task_spec(FinetuneTaskSpec *task) {
    free(task->model_name);
    free(task->optimizer);
    free(task->loss);
    free(task->metrics);
    free(task);
}

void task_append_to_json(cJSON *json, void *task_spec, MLTask ml_task) {
    switch (ml_task) {
        case T_TRAIN: {
            TrainTaskSpec *spec = (TrainTaskSpec *) task_spec;
            cJSON_AddStringToObject(json, "architecture", spec->architecture);
            cJSON_AddNumberToObject(json, "batchSize", spec->batch_size);
            cJSON_AddNumberToObject(json, "epoch", spec->epoch);
            cJSON_AddNumberToObject(json, "nBatchTrain", spec->n_batch_train);
            cJSON_AddNumberToObject(json, "nBatchEval", spec->n_batch_eval);
            cJSON_AddNumberToObject(json, "nBatchTest", spec->n_batch_test);
            cJSON_AddNumberToObject(json, "learningRate", spec->learning_rate);
            cJSON_AddStringToObject(json, "optimizer", spec->optimizer);
            cJSON_AddStringToObject(json, "loss", spec->loss);
            cJSON_AddStringToObject(json, "metrics", spec->metrics);
            break;
        }
        case T_INFERENCE: {
            InferenceTaskSpec *spec = (InferenceTaskSpec *) task_spec;
            cJSON_AddStringToObject(json, "architecture", spec->architecture);
            cJSON_AddNumberToObject(json, "batchSize", spec->batch_size);
            cJSON_AddNumberToObject(json, "nBatch", spec->n_batch);
            cJSON_AddStringToObject(json, "metrics", spec->metrics);
            break;
        }
        case T_FINETUNE: {
            FinetuneTaskSpec *spec = (FinetuneTaskSpec *) task_spec;
            cJSON_AddStringToObject(json, "modelName", spec->model_name);
            cJSON_AddNumberToObject(json, "modelId", spec->model_id);
            cJSON_AddNumberToObject(json, "batchSize", spec->batch_size);
            cJSON_AddNumberToObject(json, "epochs", spec->epochs);
            cJSON_AddNumberToObject(json, "nBatchTrain", spec->n_batch_train);
            cJSON_AddNumberToObject(json, "nBatchEval", spec->n_batch_eval);
            cJSON_AddNumberToObject(json, "nBatchTest", spec->n_batch_test);
            cJSON_AddNumberToObject(json, "learningRate", spec->learning_rate);
            cJSON_AddStringToObject(json, "optimizer", spec->optimizer);
            cJSON_AddStringToObject(json, "loss", spec->loss);
            cJSON_AddStringToObject(json, "metrics", spec->metrics);
            break;
        }
    }
}
