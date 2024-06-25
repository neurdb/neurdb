/**
 * model_inference.h
 *    provide the APIs for model load and forward inference
 */
#ifndef PG_MODEL_MODEL_INFERENCE_H
#define PG_MODEL_MODEL_INFERENCE_H

#include "../utils/torch/torch_wrapper.h"


/**
 * @description: Feed forward the input tensor through the model and get the output tensor
 * @param {ModelWrapper*} model - the model to be used
 * @param {TensorWrapper*} input - input tensor
 * @return {TensorWrapper*} - the output tensor
 */
TensorWrapper *
forward(ModelWrapper *model, TensorWrapper *input);

#endif //PG_MODEL_MODEL_INFERENCE_H
