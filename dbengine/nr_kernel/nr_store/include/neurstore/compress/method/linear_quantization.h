#ifndef LINEAR_QUANTIZATION_H
#define LINEAR_QUANTIZATION_H

#include "neurstore/utils/tensor.h"


class LinearQuantization {
public:
    template<typename QuantizedTensorType>
    static void linearSymmetricQuantize(
        const TensorType::VectorFloat64 &tensor,
        int bit_width,
        QuantizedTensorType &quantized_tensor,
        double &scale
    ) {
        // check if all elements are zero
        if (tensor.isZero()) {
            scale = 0.0f;
            quantized_tensor.resize(tensor.rows(), tensor.cols());
            quantized_tensor.setZero();
            return;
        }
        using Scalar = typename QuantizedTensorType::Scalar;
        double max_abs_val = tensor.cwiseAbs().maxCoeff();
        auto q_max = static_cast<double>((1 << (bit_width - 1)) - 1);
        scale = max_abs_val / q_max;

        quantized_tensor.resize(tensor.rows(), tensor.cols());
        quantized_tensor = (tensor.array() / scale).round().max(-q_max).min(q_max).cast<Scalar>();
    }

    template<typename QuantizedTensorType>
    static void linearSymmetricDequantize(
        const QuantizedTensorType &quantized_tensor,
        double scale,
        TensorType::VectorFloat64 &dequantized_tensor
    ) {
        dequantized_tensor.resize(quantized_tensor.rows(), quantized_tensor.cols());
        dequantized_tensor = (quantized_tensor.template cast<double>().array() * scale).matrix();
    }

    template<typename QuantizedTensorType>
    static void linearAsymmetricQuantize(
        const TensorType::VectorFloat64 &tensor,
        int bit_width,
        double min_val,
        double range, // max_val - min_val
        QuantizedTensorType &quantized_tensor,
        double &scale,
        double &zero_point
    ) {
        if (range == 0.0f) {
            scale = 1.0f;
            zero_point = -min_val;
            quantized_tensor.resize(tensor.rows(), tensor.cols());
            quantized_tensor.setZero();
            return;
        }

        #if defined(__AVX2__) && defined(__SSE4_1__)
            linearAsymmetricQuantizeAVX(tensor, bit_width, min_val, range, quantized_tensor, scale, zero_point);
        #else
            linearAsymmetricQuantizeNaive(tensor, bit_width, min_val, range, quantized_tensor, scale, zero_point);
        #endif
    }

    template<typename QuantizedTensorType>
    static void linearAsymmetricQuantize(
        const TensorType::VectorFloat64 &tensor,
        int bit_width,
        QuantizedTensorType &quantized_tensor,
        double &scale,
        double &zero_point
    ) {
        double min_val = tensor.minCoeff();
        double max_val = tensor.maxCoeff();
        double range = max_val - min_val;
        linearAsymmetricQuantize(tensor, bit_width, min_val, range, quantized_tensor, scale, zero_point);
    }

    template<typename QuantizedTensorType>
    static void linearAsymmetricQuantizeNaive(
        const TensorType::VectorFloat64 &tensor,
        int bit_width,
        double min_val,
        double range, // max_val - min_val
        QuantizedTensorType &quantized_tensor,
        double &scale,
        double &zero_point
    ) {
        // check if all elements are zero
        if (tensor.isZero()) {
            scale = 0.0f;
            quantized_tensor.resize(tensor.rows(), tensor.cols());
            quantized_tensor.setZero();
            return;
        }
        using Scalar = typename QuantizedTensorType::Scalar;
        int qmax = (1 << bit_width) - 1;
        scale = range / static_cast<double>(qmax);
        zero_point = -min_val / scale;
        Eigen::ArrayXXd array = tensor.array() / scale + zero_point;
        array = array.round().max(0.0f).min(qmax);
        quantized_tensor.resize(tensor.rows(), tensor.cols());
        quantized_tensor = array.cast<Scalar>();
    }

    template<typename QuantizedTensorType>
    static void linearAsymmetricDequantize(
        const QuantizedTensorType &quantized_tensor,
        double scale,
        double zero_point,
        TensorType::VectorFloat64 &dequantized_tensor
    ) {
        #if defined(__AVX2__) && defined(__SSE4_1__)
            linearAsymmetricDequantizeAVX(quantized_tensor, scale, zero_point, dequantized_tensor);
        #else
            linearAsymmetricDequantizeNaive(quantized_tensor, scale, zero_point, dequantized_tensor);
        #endif
    }

    template<typename QuantizedTensorType>
    static void linearAsymmetricDequantizeNaive(
        const QuantizedTensorType &quantized_tensor,
        double scale,
        double zero_point,
        TensorType::VectorFloat64 &dequantized_tensor
    ) {
        dequantized_tensor.resize(quantized_tensor.rows(), quantized_tensor.cols());

        // dequantized_val = (quantized_val - zero_point) * scale
        Eigen::ArrayXXd arr = quantized_tensor.template cast<double>().array();
        arr = (arr - zero_point) * scale;
        dequantized_tensor = arr.cast<double>().matrix();
    }

    template<typename QuantizedTensorType>
    static void linearAsymmetricDequantizeF16(
        const QuantizedTensorType &quantized_tensor,
        double scale,
        double zero_point,
        TensorType::VectorFloat16 &dequantized_tensor
    ) {
        #if defined(__AVX2__) && defined(__SSE4_1__)
                linearAsymmetricDequantizeF16AVX(quantized_tensor, scale, zero_point, dequantized_tensor);
        #else
                linearAsymmetricDequantizeF16Naive(quantized_tensor, scale, zero_point, dequantized_tensor);
        #endif
    }

    template<typename QuantizedTensorType>
    static void linearAsymmetricDequantizeF16Naive(
        const QuantizedTensorType &quantized_tensor,
        double scale,
        double zero_point,
        TensorType::VectorFloat16 &dequantized_tensor
    ) {
        dequantized_tensor.resize(quantized_tensor.rows(), quantized_tensor.cols());

        float scale_f = static_cast<float>(scale);
        float zp_f = static_cast<float>(zero_point);

        Eigen::ArrayXXf arr = quantized_tensor.template cast<float>().array();
        arr = (arr - zp_f) * scale_f;
        dequantized_tensor = arr.cast<Eigen::half>().matrix();
    }

#if defined(__AVX2__) && defined(__SSE4_1__)
    template<typename QuantizedTensorType>
    static void linearAsymmetricQuantizeAVX(
        const TensorType::VectorFloat64 &tensor,
        int bit_width,
        double min_val,
        double range, // max_val - min_val
        QuantizedTensorType &quantized_tensor,
        double &scale,
        double &zero_point
    ) {
        if (tensor.isZero()) {
            scale = 0.f;
            quantized_tensor.resize(tensor.rows(), tensor.cols());
            quantized_tensor.setZero();
            return;
        }

        using Scalar = typename QuantizedTensorType::Scalar;
        int qmax = (1 << bit_width) - 1;
        scale = range / static_cast<double>(qmax);
        zero_point = -min_val / scale;

        int n = static_cast<int>(tensor.size());
        const double *in_ptr = tensor.data();

        quantized_tensor.resize(tensor.rows(), tensor.cols());
        Scalar *out_ptr = quantized_tensor.data();

        __m256d vscale = _mm256_set1_pd(scale);
        __m256d vzeropd = _mm256_set1_pd(zero_point);
        __m256d vzero = _mm256_set1_pd(0.0);
        __m256d vmax = _mm256_set1_pd(static_cast<double>(qmax));

        int i = 0;
        const int simd = 4;
        for (; i + simd <= n; i += simd) {
            // 4 float
            __m256d vin = _mm256_loadu_pd(in_ptr + i);

            // val/scale + zero_point => multiply by 1/scale, add zero_point
            vin = _mm256_div_pd(vin, vscale);
            vin = _mm256_add_pd(vin, vzeropd);

            // clamp [0, qmax]
            vin = _mm256_max_pd(vin, vzero);
            vin = _mm256_min_pd(vin, vmax);

            // round => int32
            __m256d vrounded = _mm256_round_pd(vin, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

            // clamp in int domain
            __m128i vi_low = _mm256_cvtpd_epi32(vrounded); // 4 double ➔ 4 int32

            // cast => Scalar
            if constexpr (sizeof(Scalar) == 4) {
                _mm_storeu_si128(reinterpret_cast<__m128i *>(out_ptr + i), vi_low);
            } else {
                __m128i u16_4 = _mm_packus_epi32(vi_low, _mm_setzero_si128()); // 4 x uint16
                if constexpr (sizeof(Scalar) == 2) {
                    _mm_storel_epi64(reinterpret_cast<__m128i *>(out_ptr + i), u16_4);
                } else if constexpr (sizeof(Scalar) == 1) {
                    __m128i u8_8 = _mm_packus_epi16(u16_4, _mm_setzero_si128());
                    *(int32_t *)(out_ptr + i) = _mm_cvtsi128_si32(u8_8);
                } else {
                    throw std::runtime_error("Unsupported Scalar size");
                }
            }
        }
        // tail
        for (; i < n; i++) {
            double val = in_ptr[i] / scale + zero_point;
            val = std::round(val);
            if (val < 0.0) val = 0.0;
            if (val > (double)qmax) val = (double)qmax;
            int ival = static_cast<int>(val);

            if constexpr (sizeof(Scalar) == 1) {
                out_ptr[i] = static_cast<uint8_t>(ival);
            } else if constexpr (sizeof(Scalar) == 2) {
                out_ptr[i] = static_cast<uint16_t>(ival);
            } else {
                out_ptr[i] = static_cast<uint32_t>(ival);
            }
        }
    }


    template<typename QuantizedTensorType>
    static void linearAsymmetricDequantizeAVX(
        const QuantizedTensorType &quantized_tensor,
        float scale,
        float zero_point,
        TensorType::VectorFloat64 &dequantized_tensor
    ) {
        using Scalar = typename QuantizedTensorType::Scalar;
        dequantized_tensor.resize(quantized_tensor.rows(), quantized_tensor.cols());

        int n = static_cast<int>(quantized_tensor.size());
        const Scalar *in_ptr = quantized_tensor.data();
        double *out_ptr = dequantized_tensor.data();

        __m256d vscale = _mm256_set1_pd(scale);
        __m256d vzeropd = _mm256_set1_pd(zero_point);

        int i = 0;
        const int simd = 4;

        for (; i + simd <= n; i += simd) {
            __m256i v32;

            if constexpr (sizeof(Scalar) == 1) {
                int64_t raw8 = *(const int64_t *)(in_ptr + i);
                __m128i vi8 = _mm_cvtsi64_si128(raw8);
                __m128i zero128 = _mm_setzero_si128();
                __m128i v16_8 = _mm_unpacklo_epi8(vi8, zero128);

                __m128i v32_lo = _mm_unpacklo_epi16(v16_8, zero128);
                __m128i v32_hi = _mm_unpackhi_epi16(v16_8, zero128);
                v32 = _mm256_set_m128i(v32_hi, v32_lo);
            } else if constexpr (sizeof(Scalar) == 2) {
                __m128i v16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(in_ptr + i));
                __m128i zero128 = _mm_setzero_si128();
                __m128i v32_lo = _mm_unpacklo_epi16(v16, zero128);
                __m128i v32_hi = _mm_unpackhi_epi16(v16, zero128);
                v32 = _mm256_set_m128i(v32_hi, v32_lo);
            } else {
                v32 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in_ptr + i));
            }
            __m128i v32_low = _mm256_castsi256_si128(v32);
            __m256d vdouble = _mm256_cvtepi32_pd(v32_low);

            vdouble = _mm256_sub_pd(vdouble, vzeropd);
            vdouble = _mm256_mul_pd(vdouble, vscale);
            _mm256_storeu_pd(out_ptr + i, vdouble);
        }

        // tail
        for (; i < n; i++) {
            double v = (static_cast<double>(in_ptr[i]) - zero_point) * scale;
            out_ptr[i] = v;
        }
    }

    template<typename QuantizedTensorType>
    static void linearAsymmetricDequantizeF16AVX(
        const QuantizedTensorType& quantized_tensor,
        float scale,
        float zero_point,
        TensorType::VectorFloat16& dequantized_tensor
    ) {
        using Scalar = typename QuantizedTensorType::Scalar;
        dequantized_tensor.resize(quantized_tensor.rows(), quantized_tensor.cols());

        int n = static_cast<int>(quantized_tensor.size());
        const Scalar* in_ptr = quantized_tensor.data();
        Eigen::half* out_ptr = reinterpret_cast<Eigen::half*>(dequantized_tensor.data());

        __m256 vscale = _mm256_set1_ps(scale);
        __m256 vzp = _mm256_set1_ps(zero_point);

        int i = 0;
        const int simd = 8;  // 8x float32 (256 bit)

        for (; i + simd <= n; i += simd) {
            __m256i v32;

            if constexpr (sizeof(Scalar) == 1) {
                int64_t raw8 = *(const int64_t*)(in_ptr + i);
                __m128i vi8 = _mm_cvtsi64_si128(raw8);
                __m128i zero128 = _mm_setzero_si128();
                __m128i v16_8 = _mm_unpacklo_epi8(vi8, zero128);
                __m128i v32_lo = _mm_unpacklo_epi16(v16_8, zero128);
                __m128i v32_hi = _mm_unpackhi_epi16(v16_8, zero128);
                v32 = _mm256_set_m128i(v32_hi, v32_lo);
            } else if constexpr (sizeof(Scalar) == 2) {
                __m128i v16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in_ptr + i));
                __m128i zero128 = _mm_setzero_si128();
                __m128i v32_lo = _mm_unpacklo_epi16(v16, zero128);
                __m128i v32_hi = _mm_unpackhi_epi16(v16, zero128);
                v32 = _mm256_set_m128i(v32_hi, v32_lo);
            } else {
                v32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_ptr + i));
            }

            __m256i v32_signed = v32;
            __m256 f32 = _mm256_cvtepi32_ps(v32_signed);

            f32 = _mm256_sub_ps(f32, vzp);
            f32 = _mm256_mul_ps(f32, vscale);

            alignas(32) float tmp_f32[8];
            _mm256_store_ps(tmp_f32, f32);

            for (int j = 0; j < 8; ++j) {
                out_ptr[i + j] = static_cast<Eigen::half>(tmp_f32[j]);
            }
        }
        for (; i < n; i++) {
            float temp = (static_cast<int>(in_ptr[i]) - zero_point) * scale;
            out_ptr[i] = static_cast<Eigen::half>(temp);
        }
    }

#endif
};

#endif //LINEAR_QUANTIZATION_H
