#include "neurstore/compress/method/delta_quant_compress.h"

#include <fstream>
#include <iostream>
#include <vector>

#include "neurstore/compress/method/linear_quantization.h"
#include "neurstore/compress/utils/bit_utils.h"


/******************** DeltaQuantPacket ********************/

DeltaQuantPacket::DeltaQuantPacket(
    const TensorType::VectorUInt32 &delta_tensor,
    const double scale,
    const double zero_point,
    const int original_bit_width,
    const int quantized_bit_width,
    std::vector<int> shape
) {
    scale_ = scale;
    zero_point_ = zero_point;
    original_bit_width_ = original_bit_width;
    quantized_bit_width_ = quantized_bit_width;
    nelements_ = delta_tensor.size();
    nbytes_ = (nelements_ * quantized_bit_width + 7) / 8; // ceiling
    shape_ = std::move(shape);
    base_tensor_id_ = -1;
    serialized_data_ = BitUtil::writeBits(
        delta_tensor,
        quantized_bit_width_
    );
    index_id_ = -1;
}

int64_t DeltaQuantPacket::size() const {
    return nbytes_;
}

TensorF64 DeltaQuantPacket::toTensor64() const {
    TensorType::VectorUInt32 unpacked_delta_uint32 = BitUtil::readBits(
        serialized_data_,
        quantized_bit_width_,
        nelements_
    );
    TensorType::VectorFloat64 dequantized_delta;
    dequantized_delta.resize(nelements_);
    LinearQuantization::linearAsymmetricDequantize(
        unpacked_delta_uint32,
        scale_,
        zero_point_,
        dequantized_delta
    );
    TensorF64 tensor(
        std::move(dequantized_delta), shape_
    );
    return tensor;
}

TensorF16 DeltaQuantPacket::toTensor16() const {
    TensorType::VectorUInt32 unpacked_delta_uint32 = BitUtil::readBits(
        serialized_data_,
        quantized_bit_width_,
        nelements_
    );
    TensorType::VectorFloat16 dequantized_delta;
    dequantized_delta.resize(nelements_);
    LinearQuantization::linearAsymmetricDequantizeF16(
        unpacked_delta_uint32,
        scale_,
        zero_point_,
        dequantized_delta
    );
    TensorF16 tensor_f16(
        std::move(dequantized_delta),
        shape_
    );
    return tensor_f16;
}

UINT8QuantizedTensorPacket DeltaQuantPacket::toUINT8QuantizedTensorPacket() const {
    TensorType::VectorUInt32 unpacked_uint32 = BitUtil::readBits(
        serialized_data_,
        quantized_bit_width_,
        nelements_
    );

    // TODO: here can be optimized
    TensorType::VectorUInt8 unpacked_uint8;
    if (quantized_bit_width_ <= 8) {
        unpacked_uint8 = unpacked_uint32.unaryExpr([](uint32_t val) {
            return static_cast<uint8_t>(val & 0xFFU);
        });
    } else {
        int shift = quantized_bit_width_ - 8;
        unpacked_uint8 = unpacked_uint32.unaryExpr([shift](uint32_t val) {
            return static_cast<uint8_t>((val >> shift) & 0xFFU);
        });
    }

    UINT8QuantizedTensorPacket packet;
    packet.data = std::move(unpacked_uint8);
    packet.scale = scale_;
    packet.zero_point = zero_point_;
    packet.full_quantized_bit_width = quantized_bit_width_;
    return packet;
}

void DeltaQuantPacket::setBaseTensorId(const int64_t base_tensor_id) {
    base_tensor_id_ = base_tensor_id;
}

int64_t DeltaQuantPacket::getBaseTensorId() const {
    return base_tensor_id_;
}

std::vector<int> DeltaQuantPacket::getShape() const {
    return shape_;
}

int DeltaQuantPacket::getDimension() const {
    int dimension = 1;
    for (const auto &dim: shape_) {
        dimension *= dim;
    }
    return dimension;
}

void DeltaQuantPacket::setIndexId(int index_id) {
    this->index_id_ = index_id;
}

int DeltaQuantPacket::getIndexId() const {
    return static_cast<int>(index_id_);
}

std::vector<uint8_t> DeltaQuantPacket::serialize() const {
    const size_t shape_count = shape_.size();

    const size_t metadata_size = sizeof(scale_) + sizeof(zero_point_) + sizeof(original_bit_width_)
                                 + sizeof(quantized_bit_width_) + sizeof(base_tensor_id_) + sizeof(index_id_)
                                 + sizeof(nbytes_) + sizeof(nelements_) + sizeof(shape_count)
                                 + sizeof(int) * shape_count;

    std::vector<uint8_t> serialized_packet(metadata_size + serialized_data_.size());
    uint8_t *ptr = serialized_packet.data();

    std::memcpy(ptr, &scale_, sizeof(scale_));
    ptr += sizeof(scale_);
    std::memcpy(ptr, &zero_point_, sizeof(zero_point_));
    ptr += sizeof(zero_point_);
    std::memcpy(ptr, &original_bit_width_, sizeof(original_bit_width_));
    ptr += sizeof(original_bit_width_);
    std::memcpy(ptr, &quantized_bit_width_, sizeof(quantized_bit_width_));
    ptr += sizeof(quantized_bit_width_);
    std::memcpy(ptr, &base_tensor_id_, sizeof(base_tensor_id_));
    ptr += sizeof(base_tensor_id_);
    std::memcpy(ptr, &index_id_, sizeof(index_id_));
    ptr += sizeof(index_id_);
    std::memcpy(ptr, &nbytes_, sizeof(nbytes_));
    ptr += sizeof(nbytes_);
    std::memcpy(ptr, &nelements_, sizeof(nelements_));
    ptr += sizeof(nelements_);

    std::memcpy(ptr, &shape_count, sizeof(shape_count));
    ptr += sizeof(shape_count);
    for (const auto &dim: shape_) {
        std::memcpy(ptr, &dim, sizeof(dim));
        ptr += sizeof(dim);
    }

    std::memcpy(ptr, serialized_data_.data(), serialized_data_.size());
    return serialized_packet;
}

std::shared_ptr<TensorPacket> DeltaQuantPacket::deserialize(const std::vector<uint8_t> &data) {
    const uint8_t *ptr = data.data();
    size_t size = data.size();
    return deserialize(ptr, size);
}

std::shared_ptr<TensorPacket> DeltaQuantPacket::deserialize(const uint8_t* data, size_t size) {
    const uint8_t* ptr = data;
    double scale;
    double zero_point;
    int original_bit_width, quantized_bit_width;
    int64_t base_tensor_id, nbytes, nelements, index_id;

    std::memcpy(&scale, ptr, sizeof(scale));
    ptr += sizeof(scale);
    std::memcpy(&zero_point, ptr, sizeof(zero_point));
    ptr += sizeof(zero_point);
    std::memcpy(&original_bit_width, ptr, sizeof(original_bit_width));
    ptr += sizeof(original_bit_width);
    std::memcpy(&quantized_bit_width, ptr, sizeof(quantized_bit_width));
    ptr += sizeof(quantized_bit_width);
    std::memcpy(&base_tensor_id, ptr, sizeof(base_tensor_id));
    ptr += sizeof(base_tensor_id);
    std::memcpy(&index_id, ptr, sizeof(index_id));
    ptr += sizeof(index_id);
    std::memcpy(&nbytes, ptr, sizeof(nbytes));
    ptr += sizeof(nbytes);
    std::memcpy(&nelements, ptr, sizeof(nelements));
    ptr += sizeof(nelements);

    size_t shape_count;
    std::memcpy(&shape_count, ptr, sizeof(shape_count));
    ptr += sizeof(shape_count);

    std::vector<int> shape(shape_count);
    for (size_t i = 0; i < shape_count; ++i) {
        std::memcpy(&shape[i], ptr, sizeof(int));
        ptr += sizeof(int);
    }

    auto packet = std::make_shared<DeltaQuantPacket>(
        TensorType::VectorUInt32(),
        scale, zero_point, original_bit_width, quantized_bit_width, shape
    );
    packet->serialized_data_ = std::vector<uint8_t>(ptr, ptr + nbytes);
    packet->nbytes_ = nbytes;
    packet->nelements_ = nelements;
    packet->shape_ = std::move(shape);
    packet->base_tensor_id_ = base_tensor_id;
    packet->index_id_ = index_id;
    return packet;
}

/* For C APIs @see tensor.h */

int64_t tpt_size(const DeltaQuantPacket *packet) {
    return packet->size();
}

char *tpt_serialize(const DeltaQuantPacket *packet) {
    std::vector<uint8_t> serialized = packet->serialize();
    auto data = static_cast<char *>(malloc(serialized.size()));
    memcpy(data, serialized.data(), serialized.size());
    return data;
}

/******************** DeltaQuantCompress ********************/
DeltaQuantCompress::DeltaQuantCompress(
    double tolerance,
    bool dynamic,
    int default_quantized_bit_width
): DeltaCompress(tolerance),
   dynamic_(dynamic),
   default_quantized_bit_width_(default_quantized_bit_width) {
    if (!dynamic && default_quantized_bit_width_ < 0) {
        throw std::invalid_argument("DeltaQuantCompress: default_quantized_bit_width must be non-negative.");
    }
}


std::shared_ptr<TensorPacket> DeltaQuantCompress::compress(
    const TensorType::VectorFloat64 &base_tensor,
    const TensorType::VectorFloat64 &tensor,
    const std::vector<int> &shape
) {
    const TensorType::VectorFloat64 delta = tensor - base_tensor;

    const double delta_max = delta.maxCoeff();
    const double delta_min = delta.minCoeff();
    const auto range = delta_max - delta_min;

    int quantized_bit_width;
    if (dynamic_) {
        quantized_bit_width = BitUtil::computeBitWidth(range, this->tolerance_);
    } else {
        quantized_bit_width = default_quantized_bit_width_;
    }
    // compress
    TensorType::VectorUInt32 quantized_delta;
    double scale;
    double zero_point;
    LinearQuantization::linearAsymmetricQuantize(
        delta,
        quantized_bit_width,
        delta_min,
        range,
        quantized_delta,
        scale,
        zero_point
    );
    int original_bit_width = sizeof(float) * 8;

    return std::make_shared<DeltaQuantPacket>(
        quantized_delta,
        scale,
        zero_point,
        original_bit_width,
        quantized_bit_width,
        shape
    );
}

std::shared_ptr<TensorF64> DeltaQuantCompress::decompress(
    const TensorType::VectorFloat64 &base_tensor,
    const std::shared_ptr<TensorPacket> &t_packet
) {
    const auto delta_packet = std::dynamic_pointer_cast<DeltaQuantPacket>(t_packet);
    TensorType::VectorFloat64 delta_tensor = delta_packet->toTensor64().getTensor();
    auto tensor = std::make_shared<TensorF64>(
        base_tensor + delta_tensor, delta_packet->getShape()
    );
    return tensor;
}

std::shared_ptr<TensorF16> DeltaQuantCompress::decompressF16(
    const TensorType::VectorFloat16 &base_tensor,
    const std::shared_ptr<TensorPacket> &t_packet
) {
    const auto delta_packet = std::dynamic_pointer_cast<DeltaQuantPacket>(t_packet);
    TensorType::VectorFloat16 delta_tensor = delta_packet->toTensor16().getTensor();
    if (base_tensor.size() != delta_tensor.size()) {
        throw std::invalid_argument("DeltaQuantCompress::decompressF16: base and delta size mismatch");
    }
    TensorType::VectorFloat16 recovered_tensor(base_tensor.size());
    recovered_tensor = base_tensor + delta_tensor;
    auto tensor_f16 = std::make_shared<TensorF16>(
        recovered_tensor,
        delta_packet->getShape()
    );
    return tensor_f16;
}
