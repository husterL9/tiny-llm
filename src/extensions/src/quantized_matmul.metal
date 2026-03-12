#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

template <typename T>
[[kernel]] void quantized_matmul(device const T *x_ptr [[buffer(0)]], device const uint32_t *w_ptr [[buffer(1)]],
                                 device const T *scales_ptr [[buffer(2)]], device const T *biases_ptr [[buffer(3)]],
                                 device T *out_ptr [[buffer(4)]], constant const int &group_size_ [[buffer(5)]],
                                 constant const int &bits_ [[buffer(6)]], constant const int &M [[buffer(7)]],
                                 constant const int &N [[buffer(8)]], constant const int &K [[buffer(9)]],
                                 uint3 group_id [[threadgroup_position_in_grid]],
                                 uint3 thread_id [[thread_position_in_threadgroup]],
                                 uint3 tpg [[threads_per_threadgroup]]) {
    const int i = group_id.x * tpg.x + thread_id.x;
    const int k = group_id.y * tpg.y + thread_id.y;
    if (i >= M || k >= K) {
        return;
    }
    size_t num_groups = N / group_size_;
    size_t num_per_pack = 32 / bits_;
    size_t quantized_N = N / num_per_pack;
    size_t packs_per_group = group_size_ / num_per_pack;
    float sum = 0;
    for (size_t g = 0; g < num_groups; g++) {
        T scale = static_cast<T>(scales_ptr[k * num_groups + g]);
        T bias = static_cast<float>(biases_ptr[k * num_groups + g]);
        for (size_t index_pack = 0; index_pack < packs_per_group; index_pack++) {
            uint32_t packed_value = w_ptr[k * quantized_N + g * packs_per_group + index_pack];
            for (size_t quantized_index = 0; quantized_index < 8; ++quantized_index) {
                size_t bit_offset = quantized_index * bits_;
                uint32_t quantized = (packed_value >> bit_offset) & 0xF;
                float b_value = static_cast<float>(quantized) * scale + bias;
                size_t n = g * group_size_ + index_pack * num_per_pack + quantized_index;
                float x_value = static_cast<float>(x_ptr[i * N + n]);
                sum += x_value * b_value;
            }
        }
    }
    out_ptr[i * K + k] = static_cast<T>(sum);
}

// clang-format off
#define instantiate_quantized_matmul(type_name, type)                             \
  instantiate_kernel("quantized_matmul_" #type_name, quantized_matmul, type) 

instantiate_quantized_matmul(float16, half);
instantiate_quantized_matmul(bfloat16, bfloat16_t);
// clang-format on
