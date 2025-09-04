#include "neurstore/utils/tensor.h"

#include <memory>
#include <utility>


/*********************************** C++ ***********************************/

/* TensorWrapper */
TensorF64::TensorF64(TensorType::VectorFloat64 tensor, const std::vector<int>& shape) {
    tensor_ = std::move(tensor);
    shape_ = shape;
}

const TensorType::VectorFloat64& TensorF64::getTensor() const {
    return tensor_;
}

const std::vector<int>& TensorF64::getShape() const {
    return shape_;
}

int TensorF64::getDimension() const {
    int dimension = 1;
    for (const auto& dim : shape_) {
        dimension *= dim;
    }
    return dimension;
}

void TensorF64::setShape(const std::vector<int>& shape) {
    shape_ = shape;
}

double TensorF64::getValue(int index) const {
    return tensor_(index);
}

TensorF16::TensorF16(TensorType::VectorFloat16 tensor, const std::vector<int>& shape) {
    tensor_ = std::move(tensor);
    shape_ = shape;
}

const TensorType::VectorFloat16& TensorF16::getTensor() const {
    return tensor_;
}

const std::vector<int>& TensorF16::getShape() const {
    return shape_;
}

void TensorF16::setShape(const std::vector<int>& shape) {
    shape_ = shape;
}

int TensorF16::getDimension() const {
    int dimension = 1;
    for (const auto& dim : shape_) {
        dimension *= dim;
    }
    return dimension;
}

double TensorF16::getValue(int index) const {
    return tensor_(index);
}

IntTensorPacket::IntTensorPacket(
    IntType int_type,
    const std::vector<int> &shape,
    const std::vector<uint8_t> &serialized_data
) {
    int_type_ = int_type;
    shape_ = shape;
    nelements_ = 1;
    for (const auto &dim : shape_) {
        nelements_ *= dim;
    }
    serialized_data_ = serialized_data;
}

IntTensorPacket::IntTensorPacket(
    IntType int_type,
    const std::shared_ptr<TensorF64> &tensor
) {
    int_type_ = int_type;
    shape_ = tensor->getShape();
    nelements_ = tensor->getDimension();
    // the tensor is passed as TensorF64, but we need to convert it to the appropriate type according to int_type_
    if (int_type_ == INT8) {
        // Convert to int8
        TensorType::VectorInt8 int8_tensor = tensor->getTensor().cast<int8_t>();
        serialized_data_ = std::vector<uint8_t>(reinterpret_cast<uint8_t*>(int8_tensor.data()), reinterpret_cast<uint8_t*>(int8_tensor.data() + int8_tensor.size()));
    } else if (int_type_ == INT16) {
        TensorType::VectorInt16 int16_tensor = tensor->getTensor().cast<int16_t>();
        serialized_data_ = std::vector<uint8_t>(reinterpret_cast<uint8_t*>(int16_tensor.data()), reinterpret_cast<uint8_t*>(int16_tensor.data() + int16_tensor.size()));
    } else if (int_type_ == INT32) {
        TensorType::VectorInt32 int32_tensor = tensor->getTensor().cast<int32_t>();
        serialized_data_ = std::vector<uint8_t>(reinterpret_cast<uint8_t*>(int32_tensor.data()), reinterpret_cast<uint8_t*>(int32_tensor.data() + int32_tensor.size()));
    } else if (int_type_ == UINT8) {
        TensorType::VectorUInt8 uint8_tensor = tensor->getTensor().cast<uint8_t>();
        serialized_data_ = std::vector<uint8_t>(reinterpret_cast<uint8_t*>(uint8_tensor.data()), (uint8_tensor.data() + uint8_tensor.size()));
    } else if (int_type_ == UINT16) {
        TensorType::VectorUInt16 uint16_tensor = tensor->getTensor().cast<uint16_t>();
        serialized_data_ = std::vector<uint8_t>(reinterpret_cast<uint8_t*>(uint16_tensor.data()), reinterpret_cast<uint8_t*>(uint16_tensor.data() + uint16_tensor.size()));
    } else if (int_type_ == UINT32) {
        TensorType::VectorUInt32 uint32_tensor = tensor->getTensor().cast<uint32_t>();
        serialized_data_ = std::vector<uint8_t>(reinterpret_cast<uint8_t*>(uint32_tensor.data()), reinterpret_cast<uint8_t*>(uint32_tensor.data() + uint32_tensor.size()));
    } else {
        throw std::invalid_argument("IntTensorPacket::IntTensorPacket: Unsupported IntType for IntTensorPacket");
    }
}

int64_t IntTensorPacket::size() const {
    int type_size = getIntTypeInBytes(int_type_);
    return nelements_ * type_size;
}

TensorF64 IntTensorPacket::toTensor64() const {
    TensorType::VectorFloat64 tensor(nelements_);
    const uint8_t *data_ptr = serialized_data_.data();
    switch (int_type_) {
        case INT8: {
            const auto mapped = Eigen::Map<const TensorType::VectorInt8>(
                reinterpret_cast<const int8_t*>(data_ptr),
                nelements_
            );
            tensor = mapped.cast<double>();
            break;
        }
        case UINT8: {
            const auto mapped = Eigen::Map<const TensorType::VectorUInt8>(
                reinterpret_cast<const uint8_t*>(data_ptr),
                nelements_
            );
            tensor = mapped.cast<double>();
            break;
        }
        case INT16: {
            const auto mapped = Eigen::Map<const TensorType::VectorInt16>(
                reinterpret_cast<const int16_t*>(data_ptr),
                nelements_
            );
            tensor = mapped.cast<double>();
            break;
        }
        case UINT16: {
            const auto mapped = Eigen::Map<const TensorType::VectorUInt16>(
                reinterpret_cast<const uint16_t*>(data_ptr),
                nelements_
            );
            tensor = mapped.cast<double>();
            break;
        }
        case INT32: {
            const auto mapped = Eigen::Map<const TensorType::VectorInt32>(
                reinterpret_cast<const int32_t*>(data_ptr),
                nelements_
            );
            tensor = mapped.cast<double>();
            break;
        }
        case UINT32: {
            const auto mapped = Eigen::Map<const TensorType::VectorUInt32>(
                reinterpret_cast<const uint32_t*>(data_ptr),
                nelements_
            );
            tensor = mapped.cast<double>();
            break;
        }
        default:
            throw std::runtime_error("IntTensorPacket::toTensor64: unsupported int_type_");
    }
    return TensorF64(std::move(tensor), shape_);
}

TensorF16 IntTensorPacket::toTensor16() const {
    TensorType::VectorFloat16 tensor(nelements_);
    const uint8_t *data_ptr = serialized_data_.data();
    switch (int_type_) {
        case INT8: {
            const auto mapped = Eigen::Map<const TensorType::VectorInt8>(
                reinterpret_cast<const int8_t*>(data_ptr),
                nelements_
            );
            tensor = mapped.cast<Eigen::half>();
            break;
        }
        case UINT8: {
            const auto mapped = Eigen::Map<const TensorType::VectorUInt8>(
                reinterpret_cast<const uint8_t*>(data_ptr),
                nelements_
            );
            tensor = mapped.cast<Eigen::half>();
            break;
        }
        case INT16: {
            const auto mapped = Eigen::Map<const TensorType::VectorInt16>(
                reinterpret_cast<const int16_t*>(data_ptr),
                nelements_
            );
            tensor = mapped.cast<Eigen::half>();
            break;
        }
        case UINT16: {
            const auto mapped = Eigen::Map<const TensorType::VectorUInt16>(
                reinterpret_cast<const uint16_t*>(data_ptr),
                nelements_
            );
            tensor = mapped.cast<Eigen::half>();
            break;
        }
        case INT32: {
            const auto mapped = Eigen::Map<const TensorType::VectorInt32>(
                reinterpret_cast<const int32_t*>(data_ptr),
                nelements_
            );
            tensor = mapped.cast<Eigen::half>();
            break;
        }
        case UINT32: {
            const auto mapped = Eigen::Map<const TensorType::VectorUInt32>(
                reinterpret_cast<const uint32_t*>(data_ptr),
                nelements_
            );
            tensor = mapped.cast<Eigen::half>();
            break;
        }
        default:
            throw std::runtime_error("IntTensorPacket::toTensor16: unsupported int_type_");
    }
    return TensorF16(std::move(tensor), shape_);
}

std::vector<uint8_t> IntTensorPacket::serialize() const {
    const size_t shape_count = shape_.size();
    const size_t metadata_size = sizeof(int_type_) + sizeof(nelements_) + sizeof(shape_count) + sizeof(int) * shape_count;

    std::vector<uint8_t> serialized_packet(metadata_size + serialized_data_.size());
    uint8_t *ptr = serialized_packet.data();

    std::memcpy(ptr, &int_type_, sizeof(int_type_));
    ptr += sizeof(int_type_);
    std::memcpy(ptr, &nelements_, sizeof(nelements_));
    ptr += sizeof(nelements_);
    std::memcpy(ptr, &shape_count, sizeof(shape_count));
    ptr += sizeof(shape_count);
    for (const auto &dim : shape_) {
        std::memcpy(ptr, &dim, sizeof(int));
        ptr += sizeof(dim);
    }
    std::memcpy(ptr, serialized_data_.data(), serialized_data_.size());
    return serialized_packet;
}

void IntTensorPacket::setBaseTensorId(int64_t base_tensor_id) {
    // IntTensorPacket does not use base_tensor_id
}

int64_t IntTensorPacket::getBaseTensorId() const {
    return -1; // IntTensorPacket does not use base_tensor_id
}

std::vector<int> IntTensorPacket::getShape() const {
    return shape_;
}

int IntTensorPacket::getDimension() const {
    int dimension = 1;
    for (const auto &dim : shape_) {
        dimension *= dim;
    }
    return dimension;
}

void IntTensorPacket::setIndexId(int index_id) {
    // IntTensorPacket does not use index_id
}

int IntTensorPacket::getIndexId() const {
    return -1; // IntTensorPacket does not use index_id
}


std::shared_ptr<TensorPacket> IntTensorPacket::deserialize(const std::vector<uint8_t> &data) {
    return deserialize(data.data(), data.size());
}

std::shared_ptr<TensorPacket> IntTensorPacket::deserialize(const uint8_t *data, size_t size) {
    const uint8_t *ptr = data;
    IntType int_type;
    int64_t nelements;
    size_t shape_count;

    std::memcpy(&int_type, ptr, sizeof(int_type));
    ptr += sizeof(int_type);
    std::memcpy(&nelements, ptr, sizeof(nelements));
    ptr += sizeof(nelements);
    std::memcpy(&shape_count, ptr, sizeof(shape_count));
    ptr += sizeof(shape_count);

    std::vector<int> shape(shape_count);
    for (size_t i = 0; i < shape_count; ++i) {
        std::memcpy(&shape[i], ptr, sizeof(int));
        ptr += sizeof(int);
    }

    auto serialized_data = std::vector<uint8_t>(ptr, ptr + (size - (ptr - data)));

    return std::make_shared<IntTensorPacket>(int_type, shape, serialized_data);
}

IntTensorPacket::IntType IntTensorPacket::getIntType() const {
    return int_type_;
}

int IntTensorPacket::getIntTypeInBytes(IntTensorPacket::IntType int_type) {
    switch (int_type) {
        case IntTensorPacket::INT8:
        case IntTensorPacket::UINT8:
            return sizeof(int8_t);
        case IntTensorPacket::INT16:
        case IntTensorPacket::UINT16:
            return sizeof(int16_t);
        case IntTensorPacket::INT32:
        case IntTensorPacket::UINT32:
            return sizeof(int32_t);
        default:
            throw std::invalid_argument("getIntTypeInBytes: Unsupported IntType");
    }
}

/* TensorPacket */
/* @see delta_quant_compress.cpp */

/************************************ C ************************************/

/* TensorWrapper */
// TensorWrapperC *ts_create_tensor_wrapper(int rows, int cols) {
//     return reinterpret_cast<TensorWrapperC *>(new TensorWrapper(rows, cols));
// }
//
// void ts_destroy_tensor_wrapper(TensorWrapperC *tensor) {
//     delete reinterpret_cast<TensorWrapper *>(tensor);
// }
//
// void ts_set_value(TensorWrapperC *tensor, int row, int col, float value) {
//     reinterpret_cast<TensorWrapper *>(tensor)->setValue(row, col, value);
// }
//
// void ts_set_values(TensorWrapperC *tensor, const float *values, int n) {
//     auto tensor_ = reinterpret_cast<TensorWrapper *>(tensor);
//     long n_rows = tensor_->nRows();
//     long n_cols = tensor_->nCols();
//     if (n != n_rows * n_cols) {
//         throw std::invalid_argument("ts_set_values: n must be equal to rows * cols");
//     }
//     for (int i = 0; i < n_rows; i++) {
//         for (int j = 0; j < n_cols; j++) {
//             tensor_->setValue(i, j, values[i * n_cols + j]);
//         }
//     }
// }
//
// float ts_get_value(TensorWrapperC *tensor, int row, int col) {
//     return reinterpret_cast<TensorWrapper *>(tensor)->getValue(row, col);
// }
//
// long ts_n_rows(TensorWrapperC *tensor) {
//     return reinterpret_cast<TensorWrapper *>(tensor)->nRows();
// }
//
// long ts_n_cols(TensorWrapperC *tensor) {
//     return reinterpret_cast<TensorWrapper *>(tensor)->nCols();
// }

/* TensorPacket */
/* @see delta_quant_compress.cpp */
