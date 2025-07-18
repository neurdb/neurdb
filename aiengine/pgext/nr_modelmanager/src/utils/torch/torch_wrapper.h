/*
 * torch_wrapper.h
 *    wrapper for torchlib C++ APIs
 *    It mainly provides two wrapper classes, TensorWrapper and ModelWrapper,
 *    to pass torch::Tensor and torch::jit::script::Module between C++ and C,
 *    and related APIs to manipulate them
 */
#ifndef PG_MODEL_TORCH_WRAPPER_H
#define PG_MODEL_TORCH_WRAPPER_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/******** Struct definitions ********/
/**
 * @description: TensorWrapper is a wrapper class for torch::Tensor
 * @member {at::Tensor} tensor - the wrapped torch::Tensor
 */
typedef struct {
    void *tensor;  // torch::Tensor
} TensorWrapper;

/**
 * @description: ModelWrapper is a wrapper class for torch::jit::script::Module
 * @member {torch::jit::script::Module*} module - the wrapped
 * torch::jit::script::Module
 */
typedef struct {
    void *module;  // torch::jit::script::Module
} ModelWrapper;

/********* Functions for TensorWrapper ********/
/**
 * @description: Create a tensor from the given data (float) and dimensions
 * @param {float*} data - the data of the tensor
 * @param {int*} dims - the dimensions of the tensor, e.g., [3, 4] for a 3x4
 * matrix
 * @param {int} n_dim - the number of dimensions, e.g., 2 for a 3x4 matrix
 * @return {TensorWrapper*} - the created tensor
 */
TensorWrapper *tw_create_tensor(float *data, const int *dims, int n_dim);

/**
 * @description: Get the data of the tw_tensor
 * @param {TensorWrapper*} tw_tensor - the tw_tensor to get data from
 * @return {float*} - the data of the tw_tensor
 */
float *tw_get_tensor_data(const TensorWrapper *tw_tensor);

/**
 * @description: Get the dimensions of the tw_tensor
 * @param {TensorWrapper*} tw_tensor - the tw_tensor to get dimensions from
 * @return {int*} - the dimensions of the tw_tensor
 */
long *tw_get_tensor_dims(const TensorWrapper *tw_tensor);

/**
 * @description: Get the number of dimensions of the tw_tensor
 * @param {TensorWrapper*} tw_tensor - the tw_tensor to get the number of
 * dimensions from
 * @return {int} - the number of dimensions of the tw_tensor
 */
long tw_get_tensor_n_dim(const TensorWrapper *tw_tensor);

/********* Functions for ModelWrapper ********/
/**
 * @description: Load the model from the file system
 * @param {cstring} model_path - the path to the model
 * @return {ModelWrapper*} - the loaded model
 */
ModelWrapper *tw_load_model_by_path(const char *model_path);

/**
 * @description: Load the model from the serialized data
 * @param {cstring} model_serialized_data - the pickle serialized data of the
 * model
 * @param {size_t} size - the size of the serialized data
 * @return {ModelWrapper*} - the loaded model
 */
ModelWrapper *tw_load_model_by_serialized_data(
    const char *model_serialized_data, size_t size);

/**
 * @description: Feed tw_forward the input tensor through the tw_model and get
 * the output tensor
 * @param {ModelWrapper*} tw_model - the tw_model to be used
 * @param {TensorWrapper*} input - input tensor
 * @return {TensorWrapper*} - output tensor
 */
TensorWrapper *tw_forward(const ModelWrapper *tw_model,
                          const TensorWrapper *input);

/**
 * @description: Save the tw_model to the file system
 * Note: this function does not handle saving the tw_model to the tw_model table
 * @param {cstring} model_name - the name of the tw_model
 * @param {cstring} save_path - the path to save the tw_model
 * @param {ModelWrapper*} tw_model - the tw_model to be saved
 * @return {bool} - true if success, false otherwise
 */
bool tw_save_model(const char *model_name, const char *save_path,
                   const ModelWrapper *tw_model);

/**
 * @description: Serialize the model to a string
 * @param {ModelWrapper*} tw_model - the model to be serialized
 * @param {size_t*} size - the size of the serialized model
 * @return {cstring} - the serialized model
 */
char *tw_serialize_model(const ModelWrapper *tw_model, size_t *size);

/******** Free functions (for memory management) ********/
void tw_free_model(const ModelWrapper *tw_model);

void tw_free_tensor(const TensorWrapper *tw_tensor);

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // PG_MODEL_TORCH_WRAPPER_H
