#pragma once
#include "hnswlib.h"

namespace hnswlib {
    static float
    L2SqrQuantized(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);

        char *pVect1 = (char *) pVect1v;
        char *pVect2 = (char *) pVect2v;

        double scale1, scale2;
        double zero_point1, zero_point2;

        memcpy(&scale1, pVect1, sizeof(double));
        pVect1 += sizeof(double);
        memcpy(&zero_point1, pVect1, sizeof(double));
        pVect1 += sizeof(double);

        memcpy(&scale2, pVect2, sizeof(double));
        pVect2 += sizeof(double);
        memcpy(&zero_point2, pVect2, sizeof(double));
        pVect2 += sizeof(double);

        float scale1f, scale2f;
        float zero_point1f, zero_point2f;
        scale1f = (float) scale1;
        scale2f = (float) scale2;
        zero_point1f = (float) zero_point1;
        zero_point2f = (float) zero_point2;

        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            uint8_t q1 = *(uint8_t *) pVect1;
            uint8_t q2 = *(uint8_t *) pVect2;
            float v1 = scale1f * (float) (q1 - zero_point1);
            float v2 = scale2f * (float) (q2 - zero_point2);
            float diff = v1 - v2;
            pVect1++;
            pVect2++;
            res += diff * diff;
        }
        return (res);
    }

#if defined(USE_AVX)
    static float
    L2SqrQuantizedSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        uint8_t *pVect1 = (uint8_t *) pVect1v;
        uint8_t *pVect2 = (uint8_t *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        double scale1, scale2;
        double zp1, zp2;

        memcpy(&scale1, pVect1, sizeof(double));
        pVect1 += sizeof(double);
        memcpy(&zp1, pVect1, sizeof(double));
        pVect1 += sizeof(double);

        memcpy(&scale2, pVect2, sizeof(double));
        pVect2 += sizeof(double);
        memcpy(&zp2, pVect2, sizeof(double));
        pVect2 += sizeof(double);

        float scale1f, scale2f;
        float zp1f, zp2f;

        scale1f = (float) scale1;
        scale2f = (float) scale2;
        zp1f = (float) zp1;
        zp2f = (float) zp2;

        __m256 scaleVec1 = _mm256_set1_ps(scale1f);
        __m256 scaleVec2 = _mm256_set1_ps(scale2f);
        __m256 sum = _mm256_setzero_ps();
        size_t i = 0;

        for (; i + 16 <= qty; i += 16) {
            __m128i v1_8 = _mm_loadl_epi64((const __m128i *) (pVect1 + i));
            __m128i v2_8 = _mm_loadl_epi64((const __m128i *) (pVect2 + i));
            __m256i v1_32 = _mm256_cvtepu8_epi32(v1_8);
            __m256i v2_32 = _mm256_cvtepu8_epi32(v2_8);

            __m256 f1 = _mm256_cvtepi32_ps(v1_32);
            __m256 f2 = _mm256_cvtepi32_ps(v2_32);

            __m256 f1_adj = _mm256_sub_ps(f1, _mm256_set1_ps(zp1f));
            __m256 f2_adj = _mm256_sub_ps(f2, _mm256_set1_ps(zp2f));
            __m256 deq1 = _mm256_mul_ps(scaleVec1, f1_adj);
            __m256 deq2 = _mm256_mul_ps(scaleVec2, f2_adj);

            __m256 diff = _mm256_sub_ps(deq1, deq2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

            __m128i v1_8b = _mm_loadl_epi64((const __m128i *) (pVect1 + i + 8));
            __m128i v2_8b = _mm_loadl_epi64((const __m128i *) (pVect2 + i + 8));
            __m256i v1_32b = _mm256_cvtepu8_epi32(v1_8b);
            __m256i v2_32b = _mm256_cvtepu8_epi32(v2_8b);

            __m256 f1b = _mm256_cvtepi32_ps(v1_32b);
            __m256 f2b = _mm256_cvtepi32_ps(v2_32b);
            __m256 f1b_adj = _mm256_sub_ps(f1b, _mm256_set1_ps(zp1f));
            __m256 f2b_adj = _mm256_sub_ps(f2b, _mm256_set1_ps(zp2f));

            __m256 deq1b = _mm256_mul_ps(scaleVec1, f1b_adj);
            __m256 deq2b = _mm256_mul_ps(scaleVec2, f2b_adj);

            __m256 diffb = _mm256_sub_ps(deq1b, deq2b);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diffb, diffb));
        }

        float tmp[8];
        _mm256_storeu_ps(tmp, sum);
        return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    }
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    static DISTFUNC<float> L2SqrQuantizedSIMD16Ext = L2SqrQuantizedSIMD16ExtAVX;

    static float
    L2SqrQuantizedSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        // qty: total number of quantized elements (i.e. dimension)
        size_t qty = *((size_t *) qty_ptr);
        // Compute largest multiple of 16 less than or equal to qty.
        size_t qty16 = qty >> 4 << 4;

        // First, process the main block with the AVX function.
        float res = L2SqrQuantizedSIMD16ExtAVX(pVect1v, pVect2v, &qty16);

        // Compute pointer offset for residual part.
        // The header size (scale + zero point) is fixed.
        size_t headerSize = sizeof(double) + sizeof(double);
        const uint8_t *base1 = (const uint8_t *) pVect1v;
        const uint8_t *base2 = (const uint8_t *) pVect2v;
        // Advance to the beginning of the quantized data:
        const uint8_t *data1 = base1 + headerSize;
        const uint8_t *data2 = base2 + headerSize;

        // Advance to the residual data pointers.
        const uint8_t *residual1 = data1 + qty16;
        const uint8_t *residual2 = data2 + qty16;

        size_t qty_left = qty - qty16;

        // Re-read header information.
        double scale1, scale2;
        double zp1, zp2;
        memcpy(&scale1, base1, sizeof(double));
        zp1 = *(base1 + sizeof(double));
        memcpy(&scale2, base2, sizeof(double));
        zp2 = *(base2 + sizeof(double));

        // Process the remaining elements in a simple scalar loop.
        float res_tail = 0.0f;
        for (size_t i = 0; i < qty_left; i++) {
            int q1 = residual1[i];
            int q2 = residual2[i];
            float deq1 = (float) scale1 * ((float) q1 - (float) zp1);
            float deq2 = (float) scale2 * ((float) q2 - (float) zp2);
            float diff = deq1 - deq2;
            res_tail += diff * diff;
        }
        return res + res_tail;
    }
#endif

    class QuantizedL2Space : public SpaceInterface<float> {
        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;

    public:
        explicit QuantizedL2Space(size_t dim) {
            fstdistfunc_ = L2SqrQuantized;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
#if defined(USE_AVX512)
            if (AVX512Capable()) {
                // TODO
            }
            else if (AVXCapable()) {
                // TODO
            }
#elif defined(USE_AVX)
            if (AVXCapable())
                L2SqrQuantizedSIMD16Ext = L2SqrQuantizedSIMD16ExtAVX;
#endif
            if (dim % 16 == 0)
                fstdistfunc_ = L2SqrQuantizedSIMD16Ext;
            else if (dim > 16)
                fstdistfunc_ = L2SqrQuantizedSIMD16ExtResiduals;

#endif
            dim_ = dim;
            // scale + zero_point + data
            data_size_ = sizeof(double) + sizeof(double) + dim * sizeof(uint8_t);
        }

        size_t get_data_size() override {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() override {
            return fstdistfunc_;
        }

        void *get_dist_func_param() override {
            return &dim_;
        }

        ~QuantizedL2Space() override = default;
    };
}