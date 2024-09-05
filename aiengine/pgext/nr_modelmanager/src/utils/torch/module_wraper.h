/*
 * module_wraper.h
 *    wrapper for torchlib Module
 *
 */

#ifndef MODULE_WRAPER_H
#define MODULE_WRAPER_H
#include "torch_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/******** Struct definitions ********/
typedef struct {
  void *module;  // torch::jit::script::Module
} ModuleWrapper;

typedef struct {
  char *bytes;  // use torch::pickle::dumps to serialize
} PickledModuleWrapper;

/******** Function definitions ********/

TensorWrapper *mw_forward(const ModuleWrapper *model,
                          const TensorWrapper *input);

ModuleWrapper **mw_children(const ModuleWrapper *model, size_t *n_children);

PickledModuleWrapper *mw_pickle(const ModuleWrapper *model);

ModuleWrapper *mw_unpickle(const PickledModuleWrapper *pickled_model);

PickledModuleWrapper *mw_pickled_module_wrapper(const char *bytes, size_t size);

void mw_free_module(ModuleWrapper *model);

void mw_free_pickled_module(PickledModuleWrapper *pickled_model);

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // MODULE_WRAPER_H
