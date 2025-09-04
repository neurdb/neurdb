#include "neurstore/inference/onnx_proto_inference.h"

#include <numeric>
#include <onnxruntime_cxx_api.h>


ONNXInferenceResult ONNXProto::runInference(
    const onnx::ModelProto &model_proto,
    const TensorType::MatrixFloat32 &input_data
) {
    std::string model_data = model_proto.SerializeAsString();

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXProto");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(
        env,
        reinterpret_cast<const void *>(model_data.data()),
        model_data.size(),
        session_options
    );

    // Get the input type of the model and create the input tensor
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType input_type = input_tensor_info.GetElementType();
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor{nullptr};

    std::vector<float> input_vector_float;
    std::vector<int64_t> input_vector_int64;
    std::vector<const char *> input_vector_cstr;

    TensorType::MatrixFloat32 input_data_row_major = input_data.reshaped<Eigen::RowMajor>().eval();
    std::vector<int64_t> input_dims = {
        static_cast<int64_t>(input_data.rows()), static_cast<int64_t>(input_data.cols())
    };

    if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        // Input type is float32
        input_vector_float.assign(
            input_data_row_major.data(),
            input_data_row_major.data() + input_data_row_major.size()
        );

        input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_vector_float.data(),
            input_vector_float.size(),
            input_dims.data(),
            input_dims.size()
        );
    } else if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        // Input type is int64
        input_vector_int64.resize(input_data.size());
        std::transform(
            input_data_row_major.data(),
            input_data_row_major.data() + input_data_row_major.size(),
            input_vector_int64.begin(),
            [](float val) { return static_cast<int64_t>(val); } // Explicit conversion
        );

        input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            input_vector_int64.data(),
            input_vector_int64.size(),
            input_dims.data(),
            input_dims.size()
        );
    } else if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        // Input type is string
        std::vector<std::string> input_vector_string;
        input_vector_string.resize(input_data.size());
        std::transform(
            input_data_row_major.data(),
            input_data_row_major.data() + input_data_row_major.size(),
            input_vector_string.begin(),
            [](float val) { return std::to_string(static_cast<int>(val)); }
        );
        input_vector_cstr.reserve(input_vector_string.size());
        for (const std::string &str: input_vector_string) {
            input_vector_cstr.push_back(str.c_str());
        }
        input_tensor = Ort::Value::CreateTensor(
            allocator,
            input_dims.data(),
            input_dims.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
        );
        input_tensor.FillStringTensor(input_vector_cstr.data(), input_vector_string.size());
    } else {
        // TODO: Add support for other data types from here
        throw std::runtime_error("Unsupported input data type for the model.");
    }

    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
    const char *input_name = input_name_ptr.get();

    Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char *output_name = output_name_ptr.get();

    // perform inference
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        &input_name, &input_tensor, 1,
        &output_name, 1
    );
    auto *output_data = output_tensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> output_dims = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t total_size = std::accumulate(output_dims.begin(), output_dims.end(), 1, std::multiplies<>());
    TensorType::VectorFloat32 output(total_size);
    std::copy_n(output_data, total_size, output.data());
    ONNXInferenceResult result{output, output_dims};
    return result;
}

void ONNXInferenceResults::appenResult(const ONNXInferenceResult &result) {
    if (result.shape.size() != 2) {
        throw std::runtime_error("Invalid shape for ONNXInferenceResult: must be 2D.");
    }
    int n_samples = static_cast<int>(result.shape[0]);
    int n_outputs_per_sample = static_cast<int>(result.shape[1]);
    if (shape.empty()) {
        shape = {1, n_samples, n_outputs_per_sample};
        data.resize(n_samples * n_outputs_per_sample, 1);
        data.setZero();
    } else {
        if (shape[1] != n_samples || shape[2] != n_outputs_per_sample) {
            throw std::runtime_error("Inconsistent shape for ONNXInferenceResult.");
        }
        shape[0]++; // add model number
        data.conservativeResize(Eigen::NoChange, shape[0]);
    }
    for (int i = 0; i < data.rows(); ++i) {
        data(i, data.cols() - 1) = result.data[i];
    }
}

TensorType::MatrixFloat32 ONNXInferenceResult::toMatrix() const {
    if (shape.size() != 2) {
        throw std::runtime_error("Invalid shape for ONNXInferenceResult: must be 2D.");
    }
    return data.reshaped<Eigen::RowMajor>(shape[0], shape[1]);
}
