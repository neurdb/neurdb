#include "task.h"

#include <stdlib.h>
#include <string.h>

const char *ML_TASK[] = {"train", "inference", "finetune"};

void init_train_task_spec(TrainTaskSpec *task, const char *architecture,
                          int batch_size, int epoch, int n_batch_train,
                          int n_batch_eval, int n_batch_test,
                          double learning_rate, const char *optimizer,
                          const char *loss, const char *metrics, int cacheSize,
                          char *features, char *target, int nFeat, int nField, int nclass) {
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
  task->cacheSize = cacheSize;
  task->nFeat = nFeat;
  task->nField = nField;
  task->features = strdup(features);
  task->target = strdup(target);
  task->nclass = nclass;
}

void init_inference_task_spec(InferenceTaskSpec *task, const char *architecture,
                              int batch_size, int n_batch, const char *metrics,
                              int cacheSize, int nFeat, int nField, int nclass,
                              int modelId) {
  task->architecture = strdup(architecture);
  task->batch_size = batch_size;
  task->n_batch = n_batch;
  task->metrics = strdup(metrics);
  task->cacheSize = cacheSize;
  task->nFeat = nFeat;
  task->nField = nField;
  task->modelId = modelId;
  task->nclass = nclass;
}

void init_finetune_task_spec(FinetuneTaskSpec *task, const char *model_name,
                             int model_id, int batch_size, int epoch,
                             int n_batch_train, int n_batch_eval,
                             int n_batch_test, double learning_rate,
                             const char *optimizer, const char *loss,
                             const char *metrics, int cacheSize, int nFeat,
                             int nField) {
  task->architecture = strdup(model_name);
  task->model_id = model_id;
  task->batch_size = batch_size;
  task->epoch = epoch;
  task->n_batch_train = n_batch_train;
  task->n_batch_eval = n_batch_eval;
  task->n_batch_test = n_batch_test;
  task->learning_rate = learning_rate;
  task->optimizer = strdup(optimizer);
  task->loss = strdup(loss);
  task->metrics = strdup(metrics);
  task->cacheSize = cacheSize;
  task->nFeat = nFeat;
  task->nField = nField;
}

void free_train_task_spec(TrainTaskSpec *task) {
  free(task->architecture);
  free(task->optimizer);
  free(task->loss);
  free(task->metrics);
  free(task->features);
  free(task->target);
  free(task);
}

void free_inference_task_spec(InferenceTaskSpec *task) {
  free(task->architecture);
  free(task->metrics);
  free(task);
}

void free_finetune_task_spec(FinetuneTaskSpec *task) {
  free(task->architecture);
  free(task->optimizer);
  free(task->loss);
  free(task->metrics);
  free(task);
}

void task_append_to_json(cJSON *json, void *task_spec, MLTask ml_task) {
  switch (ml_task) {
    case T_TRAIN: {
      TrainTaskSpec *spec = (TrainTaskSpec *)task_spec;
      cJSON_AddStringToObject(json, "architecture", spec->architecture);
      cJSON_AddStringToObject(json, "features", spec->features);
      cJSON_AddStringToObject(json, "target", spec->target);
      cJSON_AddNumberToObject(json, "cacheSize", spec->cacheSize);
      cJSON_AddNumberToObject(json, "nFeat", spec->nFeat);
      cJSON_AddNumberToObject(json, "nField", spec->nField);
      cJSON_AddNumberToObject(json, "nclass", spec->nclass);

      cJSON *spec_json = cJSON_CreateObject();
      cJSON_AddNumberToObject(spec_json, "batchSize", spec->batch_size);
      cJSON_AddNumberToObject(spec_json, "epoch", spec->epoch);
      cJSON_AddNumberToObject(spec_json, "nBatchTrain", spec->n_batch_train);
      cJSON_AddNumberToObject(spec_json, "nBatchEval", spec->n_batch_eval);
      cJSON_AddNumberToObject(spec_json, "nBatchTest", spec->n_batch_test);
      cJSON_AddNumberToObject(spec_json, "learningRate", spec->learning_rate);
      cJSON_AddStringToObject(spec_json, "optimizer", spec->optimizer);
      cJSON_AddStringToObject(spec_json, "loss", spec->loss);
      cJSON_AddStringToObject(spec_json, "metrics", spec->metrics);
      cJSON_AddItemToObject(json, "spec", spec_json);
      break;
    }
    case T_INFERENCE: {
      InferenceTaskSpec *spec = (InferenceTaskSpec *)task_spec;
      cJSON_AddStringToObject(json, "architecture", spec->architecture);
      cJSON_AddNumberToObject(json, "cacheSize", spec->cacheSize);
      cJSON_AddNumberToObject(json, "nFeat", spec->nFeat);
      cJSON_AddNumberToObject(json, "nField", spec->nField);
      cJSON_AddNumberToObject(json, "nclass", spec->nclass);
      cJSON_AddNumberToObject(json, "modelId", spec->modelId);

      cJSON *spec_json = cJSON_CreateObject();
      cJSON_AddNumberToObject(spec_json, "batchSize", spec->batch_size);
      cJSON_AddNumberToObject(spec_json, "nBatch", spec->n_batch);
      cJSON_AddStringToObject(spec_json, "metrics", spec->metrics);
      cJSON_AddItemToObject(json, "spec", spec_json);
      break;
    }
    case T_FINETUNE: {
      FinetuneTaskSpec *spec = (FinetuneTaskSpec *)task_spec;
      cJSON_AddStringToObject(json, "architecture", spec->architecture);
      cJSON_AddNumberToObject(json, "modelId", spec->model_id);
      cJSON_AddNumberToObject(json, "cacheSize", spec->cacheSize);
      cJSON_AddNumberToObject(json, "nFeat", spec->nFeat);
      cJSON_AddNumberToObject(json, "nField", spec->nField);

      cJSON *spec_json = cJSON_CreateObject();
      cJSON_AddNumberToObject(spec_json, "batchSize", spec->batch_size);
      cJSON_AddNumberToObject(spec_json, "epoch", spec->epoch);
      cJSON_AddNumberToObject(spec_json, "nBatchTrain", spec->n_batch_train);
      cJSON_AddNumberToObject(spec_json, "nBatchEval", spec->n_batch_eval);
      cJSON_AddNumberToObject(spec_json, "nBatchTest", spec->n_batch_test);
      cJSON_AddNumberToObject(spec_json, "learningRate", spec->learning_rate);
      cJSON_AddStringToObject(spec_json, "optimizer", spec->optimizer);
      cJSON_AddStringToObject(spec_json, "loss", spec->loss);
      cJSON_AddStringToObject(spec_json, "metrics", spec->metrics);
      cJSON_AddItemToObject(json, "spec", spec_json);
      break;
    }
  }
}
