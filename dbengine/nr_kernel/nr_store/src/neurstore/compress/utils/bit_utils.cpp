#include "neurstore/compress/utils/bit_utils.h"

#include <immintrin.h>
#include <neurstore/utils/global.h>


std::vector<uint8_t> BitUtil::writeBits(const TensorType::VectorUInt32 &vector, const int nbits) {
#ifdef __AVX2__
    auto result = writeBitsAVX2(vector, nbits);
#else
    auto result = writeBits_naive(vector, nbits);
#endif
    return result;
}

std::vector<uint8_t> BitUtil::writeBitsNaive(
    const TensorType::VectorUInt32 &vector,
    int nbits
) {
    if (nbits < 0 || nbits > 32) {
        throw std::invalid_argument("BitUtil::writeBits: nbits must be between 0 and 32.");
    }

    const unsigned long nelements = vector.size();
    if (nelements == 0) {
        return {};
    }

    int bit_offset = 0;
    const unsigned long total_bits = nelements * nbits;
    const unsigned long total_bytes = (total_bits + 7ULL) / 8ULL; // ceiling
    auto packed_bytes = std::vector<uint8_t>(total_bytes, 0);
    for (size_t i = 0; i < nelements; ++i) {
        uint32_t val = vector.data()[i] & ((nbits == 32)
                                               ? 0xffffffffU
                                               : ((1U << nbits) - 1U));
        for (int b = 0; b < nbits; b++) {
            uint64_t this_bit = bit_offset + b;
            size_t byte_index = static_cast<size_t>(this_bit / 8ULL);
            int bit_in_byte = static_cast<int>(this_bit % 8ULL);
            if (val & (1U << b)) {
                packed_bytes[byte_index] |= static_cast<uint8_t>(1U << bit_in_byte);
            }
        }
        bit_offset += nbits;
    }
    return packed_bytes;
}

std::vector<uint8_t> BitUtil::writeBitsAVX2(
    const TensorType::VectorUInt32 &vec,
    int nbits
) {
    if (nbits < 0 || nbits > 32) {
        throw std::invalid_argument("writeBits: nbits must be in [0..32]");
    }
    const size_t nelements = vec.size();
    if (nelements == 0) {
        return {};
    }

    uint64_t total_bits = nelements * nbits;
    uint64_t total_bytes = (total_bits + 7ULL) / 8ULL;
    std::vector<uint8_t> packed(total_bytes, 0);

    int batch;
    if (nbits <= 8) batch = 8;
    else if (nbits <= 16) batch = 4;
    else batch = 2; // nbits <=32

    // mask
    uint32_t mask = (nbits == 32) ? 0xffffffffU : ((1U << nbits) - 1U);
    // offsets
    uint64_t bit_offset = 0;
    uint8_t *out_ptr = packed.data();

    __m256i vmask = _mm256_set1_epi32(static_cast<int>(mask));

    size_t i = 0;
    for (; i + batch <= nelements; i += batch) {
        alignas(32) uint32_t vals[8] = {};

        if (batch == 8) {
            // 8 x uint32
            __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(vec.data() + i)
            );
            v = _mm256_and_si256(v, vmask);
            _mm256_store_si256(reinterpret_cast<__m256i *>(vals), v);
        } else if (batch == 4) {
            // 4 x uint32
            __m128i v4 = _mm_loadu_si128(
                reinterpret_cast<const __m128i *>(vec.data() + i)
            );
            __m128i v4m = _mm_and_si128(v4, _mm_set1_epi32(static_cast<int>(mask)));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(vals), v4m);
        } else {
            // batch=2
            uint32_t v0 = vec.data()[i] & mask;
            uint32_t v1 = vec.data()[i + 1] & mask;
            vals[0] = v0;
            vals[1] = v1;
        }

        uint64_t acc = 0ULL;
        for (int k = 0; k < batch; k++) {
            acc |= ((uint64_t) vals[k]) << (k * nbits);
        }

        uint64_t old_block = 0ULL;
        size_t byte_index = bit_offset / 8ULL;
        int local_shift = static_cast<int>(bit_offset % 8ULL);

        if (byte_index + 8 <= total_bytes) {
            std::memcpy(&old_block, out_ptr + byte_index, 8);
        } else {
            size_t remain = total_bytes - byte_index;
            std::memcpy(&old_block, out_ptr + byte_index, remain);
        }

        uint64_t merged = old_block | (acc << local_shift);

        if (byte_index + 8 <= total_bytes) {
            std::memcpy(out_ptr + byte_index, &merged, 8);
        } else {
            size_t remain = total_bytes - byte_index;
            std::memcpy(out_ptr + byte_index, &merged, remain);
        }
        bit_offset += batch * nbits;
    }

    for (; i < nelements; i++) {
        uint32_t v = (vec.data()[i] & mask);
        for (int b = 0; b < nbits; b++) {
            uint64_t this_bit = bit_offset + b;
            size_t byte_index = (size_t) (this_bit / 8ULL);
            int bit_in_byte = static_cast<int>(this_bit % 8ULL);
            if (v & (1U << b)) {
                packed[byte_index] |= (uint8_t) (1U << bit_in_byte);
            }
        }
        bit_offset += nbits;
    }
    return packed;
}

TensorType::VectorUInt32 BitUtil::readBits(
    const std::vector<uint8_t> &packed_bytes,
    int nbits,
    unsigned long nelements
) {
#ifdef __AVX2__
    auto result = readBitsAVX2(packed_bytes, nbits, nelements);
#else
    auto result = readBits_naive(packed_bytes, nbits, nelements);
#endif
    return result;
}

TensorType::VectorUInt32 BitUtil::readBitsAVX2(
    const std::vector<uint8_t> &packed_bytes,
    int nbits,
    unsigned long nelements
) {
    if (nbits < 0 || nbits > 32) {
        throw std::invalid_argument("readBitsAVX2: nbits must be in [0..32]");
    }
    TensorType::VectorUInt32 result(nelements);
    if (nelements == 0) return result;

    uint64_t total_bits = nelements * nbits;
    uint64_t total_bytes = (total_bits + 7ULL) / 8ULL;
    if (packed_bytes.size() < total_bytes) {
        throw std::runtime_error("readBitsAVX2: input buffer too small");
    }

    int batch;
    if (nbits <= 8) batch = 8;
    else if (nbits <= 16) batch = 4;
    else batch = 2;

    uint32_t mask = (nbits == 32) ? 0xffffffffU : ((1U << nbits) - 1U);
    uint64_t bit_offset = 0ULL;

    size_t i = 0;
    for (; i + batch <= nelements; i += batch) {
        uint64_t block = 0ULL;
        size_t byte_index = bit_offset / 8ULL;
        int local_shift = static_cast<int>(bit_offset % 8ULL);

        if (byte_index + 8 <= packed_bytes.size()) {
            std::memcpy(&block, &packed_bytes[byte_index], 8);
        } else {
            size_t remain = packed_bytes.size() - byte_index;
            std::memcpy(&block, &packed_bytes[byte_index], remain);
        }

        uint64_t acc = (block >> local_shift);
        alignas(32) uint32_t vals[8];
        for (int k = 0; k < batch; k++) {
            vals[k] = static_cast<uint32_t>((acc >> (k * nbits)) & mask);
        }

        if (batch == 8) {
            __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i *>(vals));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(result.data() + i), v);
        } else if (batch == 4) {
            __m128i v4 = _mm_load_si128(reinterpret_cast<const __m128i *>(vals));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(result.data() + i), v4);
        } else {
            result.data()[i] = vals[0];
            result.data()[i + 1] = vals[1];
        }

        bit_offset += static_cast<uint64_t>(batch) * nbits;
    }

    // tail: naive
    for (; i < nelements; i++) {
        uint32_t val = 0U;
        for (int b = 0; b < nbits; b++) {
            uint64_t this_bit = bit_offset + b;
            size_t byte_i = static_cast<size_t>(this_bit / 8ULL);
            int bit_in_byte = static_cast<int>(this_bit % 8ULL);
            if (packed_bytes[byte_i] & (1U << bit_in_byte)) {
                val |= (1U << b);
            }
        }
        result.data()[i] = val;
        bit_offset += nbits;
    }
    return result;
}

TensorType::VectorUInt32 BitUtil::readBitsNaive(
    const std::vector<uint8_t> &packed_bytes,
    const int nbits,
    const unsigned long nelements
) {
    if (nbits < 0 || nbits > 32) {
        throw std::invalid_argument("BitUtil::readBitsNaive: nbits must be between 0 and 32.");
    }

    TensorType::VectorUInt32 unpacked_array(nelements);

    uint64_t total_bits = static_cast<uint64_t>(nelements) * nbits;
    uint64_t total_bytes = (total_bits + 7ULL) / 8ULL;
    if (packed_bytes.size() < total_bytes) {
        throw std::runtime_error("BitUtil::readBitsNaive: packed_bytes is too small for the given nbits*nelements");
    }

    uint64_t bit_offset = 0;
    for (size_t i = 0; i < nelements; i++) {
        uint32_t value = 0;
        for (int b = 0; b < nbits; b++) {
            uint64_t this_bit = bit_offset + b;
            size_t byte_index = static_cast<size_t>(this_bit / 8ULL);
            int bit_in_byte = static_cast<int>(this_bit % 8ULL);

            if (packed_bytes[byte_index] & (1U << bit_in_byte)) {
                value |= (1U << b);
            }
        }
        unpacked_array.data()[i] = value;
        bit_offset += nbits;
    }
    return unpacked_array;
}

int BitUtil::computeBitWidth(const double range, const double tolerance) {
    if (range == 0.0f) {
        return 1;
    }
    const int bit_width = static_cast<int>(std::ceil(std::log2(range / (tolerance * 2.0f))));
    return std::max(1, std::min(bit_width, 32));
}
