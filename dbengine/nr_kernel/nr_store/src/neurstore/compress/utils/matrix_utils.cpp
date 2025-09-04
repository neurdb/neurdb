#include "neurstore/compress/utils/matrix_utils.h"


TensorF64 MatrixUtils::tensorProto2TensorF64(const onnx::TensorProto &tensor_proto) {
    if (tensor_proto.raw_data().empty()) {
        throw std::runtime_error("MatrixUtils::tensorProto2TensorF64: tensor_proto does not contain raw data");
    }
    size_t total_size = 1;
    std::vector<int> shape;
    for (int i = 0; i < tensor_proto.dims_size(); ++i) {
        total_size *= tensor_proto.dims(i);
        shape.push_back(static_cast<int>(tensor_proto.dims(i)));
    }

    TensorType::VectorFloat64 float64_vector;

    if (tensor_proto.data_type() == onnx::TensorProto::FLOAT) {
        const auto *raw_data_ptr = reinterpret_cast<const float *>(tensor_proto.raw_data().data());
        Eigen::Map<const TensorType::VectorFloat32> raw_vector(
            raw_data_ptr,
            static_cast<Eigen::Index>(total_size)
        );
        float64_vector = raw_vector.cast<double>();
    } else if (tensor_proto.data_type() == onnx::TensorProto::FLOAT16) {
        const auto *raw_data_ptr = reinterpret_cast<const Eigen::half *>(tensor_proto.raw_data().data());
        Eigen::Map<const TensorType::VectorFloat16> raw_vector(
            raw_data_ptr,
            static_cast<Eigen::Index>(total_size)
        );
        float64_vector = raw_vector.cast<float>().cast<double>();
    } else if (tensor_proto.data_type() == onnx::TensorProto::DOUBLE) {
        const auto *raw_data_ptr = reinterpret_cast<const double *>(tensor_proto.raw_data().data());
        Eigen::Map<const TensorType::VectorFloat64> raw_vector(
            raw_data_ptr,
            static_cast<Eigen::Index>(total_size)
        );
        float64_vector = raw_vector;
    } else {
        throw std::runtime_error("MatrixUtils::tensorProto2TensorF64: unsupported data type");
    }

    TensorF64 tensor(std::move(float64_vector), shape);
    return tensor;
}
