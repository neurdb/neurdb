#ifndef HTTP_H
#define HTTP_H

#include "interface.h"

/**
 * Send a training task to the server
 * @param info TrainingInfo* Training task info
 */
void *send_train_task(TrainingInfo *info);

/**
 * Resquest the server to make a forward inference with a model
 * @param info InferenceInfo* Inference task info
 */
void *send_inference_task(InferenceInfo *info);

/**
 * Resquest the server to finetune a model
 * @param info FinetuneInfo* Finetune task info
 */
void *send_finetune_task(FinetuneInfo *info);
#endif
