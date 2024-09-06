#include "module_wraper.h"

#include <torch/torch.h>
#include <torch/script.h>

TensorWrapper *mw_forward(const ModuleWrapper *model,
                          const TensorWrapper *input) {
  const auto module = static_cast<torch::jit::script::Module *>(model->module);
  const auto tensor_input = static_cast<torch::Tensor *>(input->tensor);
  // forward inference
  try {
    const auto output = module->forward({*tensor_input}).toTensor();
    const auto tw_output = new TensorWrapper();
    tw_output->tensor = new torch::Tensor(output);
    return tw_output;
  } catch (const std::runtime_error &e) {
    std::cerr << "Error during forward inference: " << e.what() << std::endl;
    return nullptr;
  }
}

ModuleWrapper **mw_children(const ModuleWrapper *mw_module,
                            size_t *n_children) {
  const auto module =
      static_cast<torch::jit::script::Module *>(mw_module->module);
  const auto children = module->children();
  *n_children = children.size();
  const auto children_array = new ModuleWrapper *[*n_children];

  size_t index = 0;
  for (const auto &child : children) {
    const auto childModule = new ModuleWrapper();
    childModule->module = new torch::jit::script::Module(
        child);  // this is done to ensure deep copy
    children_array[index++] = childModule;
  }
  return children_array;
}

PickledModuleWrapper *mw_pickle(const ModuleWrapper *model) {
  torch::jit::script::Module *module =
      static_cast<torch::jit::script::Module *>(model->module);
  std::ostringstream output_stream;
  // torch::jit::pickle_save(*module, &output_stream);
}

ModuleWrapper *mw_unpickle(const PickledModuleWrapper *pickled_model) {
  // TODO: implement this
  return NULL;
}

PickledModuleWrapper *mw_pickled_module_wrapper(const char *bytes,
                                                size_t size) {
  // TODO: implement this
  return NULL;
}

void mw_free_module(ModuleWrapper *model) {
  if (model) {
    const auto module =
        static_cast<torch::jit::script::Module *>(model->module);
    delete module;
    model->module = nullptr;
    delete model;
  }
}

void mw_free_children(ModuleWrapper **model, size_t n_children) {
  for (size_t i = 0; i < n_children; i++) {
    mw_free_module(model[i]);
  }
  delete[] model;
}

void mw_free_pickled_module(PickledModuleWrapper *pickled_model) {
  // TODO: implement this
}
