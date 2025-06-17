/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

 #ifndef UTILS_CUH_
 #define UTILS_CUH_
 
 
 #include <cuda_bf16.h>
 #include <cuda_fp16.h>
 #include <cuda_fp8.h>
 #include <cstdint>
 
 
 
 using fp8e4m3 = __nv_fp8_e4m3;
 using fp8e5m2 = __nv_fp8_e5m2;
 using e8m0_t = uint8_t;
 
 constexpr uint32_t FP32_MANTISSA_BITS = 23;
 constexpr uint32_t FP32_EXPONENT_BIAS = 127;
 
 template <typename T>
 struct Numeric_Traits;
 
 template <>
 struct Numeric_Traits<fp8e4m3> {
   static constexpr int maxUnbiasedExponent = 8;
   static constexpr double maxNorm = 448;
 };
 
 template <>
 struct Numeric_Traits<fp8e5m2> {
   static constexpr int maxUnbiasedExponent = 15;
   static constexpr double maxNorm = 57344;
 };
 
 
 
 
 template <typename T>
 struct Quantized_Limits {
   static constexpr int max_unbiased_exponent = Numeric_Traits<T>::maxUnbiasedExponent;
   static constexpr float max_norm = Numeric_Traits<T>::maxNorm;
   static constexpr float max_norm_rcp = 1.0 / max_norm;
   static constexpr float emax = 1 << max_unbiased_exponent;
   static constexpr float emax_rcp = 1.0 / emax;
 };
 
 __device__ __forceinline__ e8m0_t float_to_e8m0(float val) {
   // TODO: nan/inf needs to be set for any value
   // of nan/inf in input not just amax.
   if (isnan(val)) {
     return 0xFF;
   }
   if (isinf(val)) {
     return 0xFE;
   }
 #if ((__CUDA_ARCH_HAS_FEATURE__(SM100_ALL)) || (__CUDA_ARCH_HAS_FEATURE__(SM101_ALL)) || \
      (__CUDA_ARCH_HAS_FEATURE__(SM120_ALL)))
   uint16_t out;
   asm volatile(
       "{\n"
       "cvt.rp.satfinite.ue8m0x2.f32  %0, 0.0, %1;\n"
       "}"
       : "=h"(out)
       : "f"(val));
   return *reinterpret_cast<e8m0_t *>(&out);
 #else
   if (val == 0.0f) {
     return 0x00;
   }
   uint32_t val_u32 = *reinterpret_cast<uint32_t *>(&val);
   e8m0_t exponent = (val_u32 >> FP32_MANTISSA_BITS);
   uint32_t mantissa = val_u32 & 0x7FFFFF;
   // Round up exponent and deal with satfinite.
   if ((mantissa > 0 && exponent != 0xFE) && !(exponent == 0 && mantissa <= 0x400000)) {
     ++exponent;
   }
   return exponent;
 #endif
 }
 
 
 __device__ __forceinline__ float exp2f_rcp(e8m0_t biased_exp) {
   return (biased_exp == 0) ? 1 : exp2f(FP32_EXPONENT_BIAS - static_cast<float>(biased_exp));
 }
 
 #endif
