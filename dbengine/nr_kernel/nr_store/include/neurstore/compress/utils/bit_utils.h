#ifndef BIT_UTILS_H
#define BIT_UTILS_H

#include <cstdint>
#include <vector>

#include "neurstore/utils/tensor.h"


/**
 * BitUtil provides utility functions for reading and writing tensors in bits.
 * This allows applying unstandardized bit-width in compression algorithms.
 *
 * @note We use uint32_t as the basic unit for reading and writing bits to
 * guarantee enough space for all possible bit-widths.
 */
class BitUtil {
public:
    /**
     * Utility function to write a tensor in bits.
     * @param vector The tensor to write, it has to be a tensor of shape (nelements, 1)
     * @param nbits The number of bits to use for each element
     * @return The serialized tensor
     */
    static std::vector<uint8_t> writeBits(const TensorType::VectorUInt32 &vector, int nbits);

    /**
     * Utility function to read a tensor from bits.
     * @param packed_bytes Serialized tensor
     * @param nbits The number of bits used for each element
     * @param nelements The number of elements in the tensor
     * @return The deserialized tensor
     */
    static TensorType::VectorUInt32 readBits(
        const std::vector<uint8_t> &packed_bytes,
        int nbits,
        unsigned long nelements
    );

    /**
     * Utility function to compute the minimum number of bits required to serialize each
     * element of a float tensor.
     * @param range The range of the tensor
     * @param tolerance The tolerance in the quantization
     * @return
     */
    static int computeBitWidth(double range, double tolerance);

private:
#ifdef __AVX2__
    static std::vector<uint8_t> writeBitsAVX2(
        const TensorType::VectorUInt32 &vec,
        int nbits
    );

    static TensorType::VectorUInt32 readBitsAVX2(
        const std::vector<uint8_t> &packed_bytes,
        int nbits,
        unsigned long nelements
    );
#endif

    static std::vector<uint8_t> writeBitsNaive(
        const TensorType::VectorUInt32 &vector,
        int nbits
    );

    static TensorType::VectorUInt32 readBitsNaive(
        const std::vector<uint8_t> &packed_bytes,
        int nbits,
        unsigned long nelements
    );
};

#endif //BIT_UTILS_H
