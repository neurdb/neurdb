#include "torch_wrapper.h"

#include <torch/torch.h>
#include <torch/script.h>

#include "device.h"

TensorWrapper* tw_create_tensor(float* data, const int* dims, const int n_dim) {
    const std::vector<int64_t> dims_int64(
        dims, dims + n_dim);  // compose the dimensions
    const auto tw_tensor = new TensorWrapper();
    // ArrayRef<int64_t> is a reference to the dimensions of the tensor
    const auto tensor = new torch::Tensor(
        torch::from_blob(data, at::ArrayRef<int64_t>(dims_int64), at::kFloat));
    tensor->to(*static_cast<torch::Device*>(device));  // move tensor to device
    tw_tensor->tensor = tensor;
    return tw_tensor;
}

float* tw_get_tensor_data(const TensorWrapper* tw_tensor) {
    const auto tensor = static_cast<torch::Tensor*>(
        tw_tensor->tensor);  // cast void* to torch::Tensor*
    return tensor->data_ptr<float>();
}

long* tw_get_tensor_dims(const TensorWrapper* tw_tensor) {
    const auto tensor = static_cast<torch::Tensor*>(tw_tensor->tensor);
    const auto sizes = tensor->sizes();
    const auto dims = new long[sizes.size()];
    for (int i = 0; i < sizes.size(); i++) {
        dims[i] = sizes[i];
    }
    return dims;
}

long tw_get_tensor_n_dim(const TensorWrapper* tw_tensor) {
    // dim() returns the number of dimensions, yet sizes() returns the sizes of
    // each dimension
    // @see: https://pytorch.org/docs/stable/generated/torch.Tensor.size.html
    const auto tensor = static_cast<torch::Tensor*>(tw_tensor->tensor);
    return tensor->dim();
}

ModelWrapper* tw_load_model_by_path(const char* model_path) {
    const auto tw_model = new ModelWrapper();
    const auto module =
        new torch::jit::script::Module(torch::jit::load(model_path));
    module->to(*static_cast<torch::Device*>(device));  // move module to device
    tw_model->module = module;
    return tw_model;
}

ModelWrapper* tw_load_model_by_serialized_data(
    const char* model_serialized_data, const size_t size) {
    const auto tw_model = new ModelWrapper();
    std::istringstream input_stream(std::string(model_serialized_data, size));
    const auto module = torch::jit::load(input_stream);

    const auto script_module = new torch::jit::script::Module(module);
    script_module->to(
        *static_cast<torch::Device*>(device));  // move module to device
    tw_model->module = script_module;
    return tw_model;
}

TensorWrapper* tw_forward(const ModelWrapper* tw_model,
                          const TensorWrapper* input) {
    const auto module =
        static_cast<torch::jit::script::Module*>(tw_model->module);
    const auto tensor_input = static_cast<torch::Tensor*>(input->tensor);
    module->to(*static_cast<torch::Device*>(device));  // move module to device
    tensor_input->to(
        *static_cast<torch::Device*>(device));  // move tensor to device
    // forward inference
    try {
        const auto output = module->forward({*tensor_input}).toTensor();
        const auto tw_output = new TensorWrapper();
        tw_output->tensor = new torch::Tensor(output);
        return tw_output;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error during forward inference: " << e.what()
                  << std::endl;
        return nullptr;
    }
}

bool tw_save_model(const char* model_name, const char* save_path,
                   const ModelWrapper* tw_model) {
    // concatenate the save path and tw_model name
    const std::string full_save_path =
        std::string(save_path) + "/" + std::string(model_name) + ".pt";
    try {
        const auto module =
            static_cast<torch::jit::script::Module*>(tw_model->module);
        module->save(full_save_path);
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "tw_save_model: error saving tw_model: " << e.what()
                  << std::endl;
        return false;
    }
}

char* tw_serialize_model(const ModelWrapper* tw_model, size_t* size) {
    const auto module = static_cast<torch::jit::script::Module*>(
        tw_model->module);  // the model to be serialized
    // save the model to a stringstream buffer
    std::stringstream ss;
    module->save(ss);
    const std::string serialized_data = ss.str();
    *size = serialized_data.size();

    const auto data = static_cast<char*>(std::malloc(*size));
    if (!data) {
        // throw an exception if memory allocation fails
        throw std::bad_alloc();
    }
    std::memcpy(data, serialized_data.data(), *size);
    return data;
}

void tw_free_model(const ModelWrapper* tw_model) {
    if (tw_model) {
        delete static_cast<torch::jit::script::Module*>(tw_model->module);
        delete tw_model;
    }
}

void tw_free_tensor(const TensorWrapper* tw_tensor) {
    if (tw_tensor) {
        delete static_cast<torch::Tensor*>(tw_tensor->tensor);
        delete tw_tensor;
    }
}
