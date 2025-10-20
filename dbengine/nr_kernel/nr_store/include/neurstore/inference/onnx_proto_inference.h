#ifndef ONNX_PROTO_INFERENCE_H
#define ONNX_PROTO_INFERENCE_H

#include <onnx.pb.h>

#include "neurstore/utils/tensor.h"

struct ONNXInferenceResult {
    TensorType::VectorFloat32 data;
    std::vector<int64_t> shape;

    TensorType::MatrixFloat32 toMatrix() const;
};

struct ONNXInferenceResults {
    TensorType::MatrixFloat32 data;
    std::vector<int64_t> shape;

    void appenResult(const ONNXInferenceResult &result);
};

class ONNXProto {
public:
    /**
     * Perform inference using ONNX model
     * @param model_proto The ONNX ModelProto object
     * @param input_data The input data
     * @return The output data
     */
    static ONNXInferenceResult runInference(
        const onnx::ModelProto &model_proto,
        const TensorType::MatrixFloat32 &input_data
    );
};

#endif //ONNX_PROTO_INFERENCE_H
