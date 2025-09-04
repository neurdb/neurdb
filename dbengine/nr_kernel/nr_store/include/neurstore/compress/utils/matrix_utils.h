#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <onnx.pb.h>

#include "neurstore/utils/tensor.h"


/**
 * MatrixUtils provides utility functions for Eigen matrices and onnx tensors.
 */
class MatrixUtils {
public:
    static TensorF64 tensorProto2TensorF64(const onnx::TensorProto &tensor_proto);
};

#endif //MATRIX_UTILS_H
