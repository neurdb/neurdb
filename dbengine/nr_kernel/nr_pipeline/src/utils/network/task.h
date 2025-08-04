#ifndef TASK_H
#define TASK_H

#include <cjson/cJSON.h>

extern const char *ML_TASK[];

/**
 * The machine learning task
 * T_TRAIN: Train the model
 * T_INFERENCE: Inference with the model
 * T_FINETUNE: Finetune the model
 */
typedef enum { T_TRAIN = 0, T_INFERENCE = 1, T_FINETUNE = 2 } MLTask;

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

  int cacheSize;
  char *features;
  char *target;
  int nFeat;
  int nField;

  int nclass;
} TrainTaskSpec;

typedef struct {
  char *architecture;
  int batch_size;
  int n_batch;
  char *metrics;

  int cacheSize;
  char *features;
  char *target;
  int nFeat;
  int nField;
  int modelId;

  int nclass;
} InferenceTaskSpec;

typedef struct {
  char *architecture;
  int model_id;
  int batch_size;
  int epoch;
  int n_batch_train;
  int n_batch_eval;
  int n_batch_test;
  double learning_rate;
  char *optimizer;
  char *loss;
  char *metrics;

  int cacheSize;
  int nFeat;
  int nField;
} FinetuneTaskSpec;

void init_train_task_spec(TrainTaskSpec *task, const char *architecture,
                          int batch_size, int epoch, int n_batch_train,
                          int n_batch_eval, int n_batch_test,
                          double learning_rate, const char *optimizer,
                          const char *loss, const char *metrics, int cacheSize,
                          char *features, char *target, int nFeat, int nField, int nclass);

void init_inference_task_spec(InferenceTaskSpec *task, const char *architecture,
                              int batch_size, int n_batch, const char *metrics,
                              int cacheSize, int nFeat, int nField, int nclass,
                              int modelId, char *features, char *target);

void init_finetune_task_spec(FinetuneTaskSpec *task, const char *model_name,
                             int model_id, int batch_size, int epoch,
                             int n_batch_train, int n_batch_eval,
                             int n_batch_test, double learning_rate,
                             const char *optimizer, const char *loss,
                             const char *metrics, int cacheSize, int nFeat,
                             int nField);

void free_train_task_spec(TrainTaskSpec *task);

void free_inference_task_spec(InferenceTaskSpec *task);

void free_finetune_task_spec(FinetuneTaskSpec *task);

/**
 * Append the task specification to the json object
 * @param json The json object to be appended
 * @param task_spec The task specification, it can be TrainTaskSpec,
 * InferenceTaskSpec, or FinetuneTaskSpec
 * @param ml_task The machine learning task
 */
void task_append_to_json(cJSON *json, void *task_spec, MLTask ml_task);

#endif  // TASK_H
