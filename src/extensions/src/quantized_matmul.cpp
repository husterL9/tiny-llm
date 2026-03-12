#include <mlx/dtype.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>

#include "axpby.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext {
mx::array quantized_matmul(const mx::array &x,       // M × N (float16, activations)
                           const mx::array &weight,  // Input array K × (N/8) (uint32, packed weights)
                           const mx::array &scales, const mx::array &biases, int group_size, int bits, bool transpose_b,
                           mx::StreamOrDevice s  // Stream on which to schedule the operation
) {
    if (!transpose_b) {
        throw std::runtime_error("quantized_matmul: b must be transposed");
    }
    auto out_shape = mx::Shape{x.shape(0), weight.shape(0)};
    std::vector<mx::array> inputs = {x, weight, scales, biases};
    return mx::array(
        /* const mx::Shape& shape = */ out_shape,
        /* mx::Dtype dtype = */ x.dtype(),
        /* std::shared_ptr<mx::Primitive> primitive = */
        std::make_shared<QuantizedMatmul>(to_stream(s), group_size, bits),
        /* const std::vector<mx::array>& inputs = */ inputs);
}
template <typename T>
void quantized_matmul_impl(const mx::array &x, const mx::array &weight, const mx::array &scales,
                           const mx::array &biases, mx::array &out, int group_size_, int bits_, mx::Stream stream) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    // Get the CPU command encoder and register input and output arrays
    auto &encoder = mx::cpu::get_command_encoder(stream);
    encoder.set_input_array(x);
    encoder.set_input_array(weight);
    encoder.set_input_array(scales);
    encoder.set_input_array(biases);
    encoder.set_output_array(out);

    // Launch the CPU kernel
    encoder.dispatch([x_ptr = x.data<T>(), w_ptr = weight.data<uint32_t>(), scales_ptr = scales.data<T>(),
                      biases_ptr = biases.data<T>(), out_ptr = out.data<T>(), size = out.size(), shape = out.shape(),
                      x_shape = x.shape(), x_strides = x.strides(), w_strides = weight.strides(), group_size_,
                      bits_]() {
        size_t num_per_pack = 32 / bits_;
        size_t M = shape[0];
        size_t K = shape[1];
        size_t N = x_shape[1];
        size_t quantized_N = N / num_per_pack;
        size_t num_groups = N / group_size_;

        size_t packs_per_group = group_size_ / num_per_pack;
        // Do the matmul operation for each output(i,k)
        for (size_t out_idx = 0; out_idx < size; out_idx++) {
            float sum = 0;
            size_t i = out_idx / K;
            size_t k = out_idx % K;
            for (size_t g = 0; g < num_groups; g++) {
                float scale = static_cast<float>(scales_ptr[k * num_groups + g]);
                float bias = static_cast<float>(biases_ptr[k * num_groups + g]);
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
            out_ptr[out_idx] = sum;
        }
    });
}
void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    auto &x = inputs[0];
    auto &weight = inputs[1];
    auto &scales = inputs[2];
    auto &biases = inputs[3];

    auto &out = outputs[0];
    switch (x.dtype()) {
        case mx::float16:
            quantized_matmul_impl<mx::float16_t>(x, weight, scales, biases, out, group_size_, bits_, stream());
            break;
        case mx::float32:
            quantized_matmul_impl<float>(x, weight, scales, biases, out, group_size_, bits_, stream());
            break;
        case mx::bfloat16:
            quantized_matmul_impl<mx::bfloat16_t>(x, weight, scales, biases, out, group_size_, bits_, stream());
            break;
        default:
            throw std::runtime_error("Unsupported dtype for quantized_matmul");
    }
};
void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    // Prepare inputs
    auto &x = inputs[0];
    auto &weight = inputs[1];
    auto &scales = inputs[2];
    auto &biases = inputs[3];
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    size_t nelem = out.size();

    // Each primitive carries the stream it should execute on
    // and each stream carries its device identifiers
    auto &s = stream();
    // We get the needed metal device using the stream
    auto &d = mx::metal::device(s.device);

    // Resolve name of kernel (corresponds to axpby.metal)
    std::ostringstream kname;
    kname << "quantized_matmul_";
    kname << type_to_name(out);
    // std::cout << "quantized_matmul kernel: " << kname.str() << std::endl;
    // Make a kernel from this metal library (use lib name overload)
    auto library = d.get_library("tiny_llm_ext");
    auto kernel = d.get_kernel(kname.str(), library);

    // Prepare to encode kernel
    auto &compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);

    // Encode input arrays to kernel
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_input_array(weight, 1);
    compute_encoder.set_input_array(scales, 2);
    compute_encoder.set_input_array(biases, 3);

    // Encode output arrays to kernel
    compute_encoder.set_output_array(out, 4);

    // Encode group_size_ and bits_
    compute_encoder.set_bytes(group_size_, 5);
    compute_encoder.set_bytes(bits_, 6);

    // We launch 1 thread for each input and make sure that the number of
    // threads in any given threadgroup is not higher than the max allowed
    size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();
    const int tile_x = 32;
    const int tile_y = tgp_size / tile_x;
    // Fix the 3D size of each threadgroup (in terms of threads)
    MTL::Size group_dims = MTL::Size(tile_x, tile_y, 1);
    size_t M = x.shape()[0];
    size_t N = x.shape()[1];
    size_t K = weight.shape()[0];

    compute_encoder.set_bytes(M, 7);
    compute_encoder.set_bytes(N, 8);
    compute_encoder.set_bytes(K, 9);
    // Fix the 3D size of the launch grid (in terms of threads)
    MTL::Size grid_dims = MTL::Size((M + tile_x - 1) / tile_x, (K + tile_y - 1) / tile_y, 1);

    // Launch the grid with the given number of threads divided among
    // the given threadgroups
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

/** Print primitive name and parameters */
void QuantizedMatmul::print(std::ostream &os) {
    os << name() << "(group_size=" << group_size_ << ", bits=" << bits_ << ")";
}
/** Vectorize primitive along given axis */
std::pair<std::vector<mx::array>, std::vector<int>> QuantizedMatmul::vmap(const std::vector<mx::array> &inputs,
                                                                          const std::vector<int> &axes) {
    throw std::runtime_error("QuantizedMatmul has no vmap implementation.");
}

/** Equivalence check **/
bool QuantizedMatmul::is_equivalent(const Primitive &other) const {
    const QuantizedMatmul &r_other = static_cast<const QuantizedMatmul &>(other);
    return group_size_ == r_other.group_size_ && bits_ == r_other.bits_;
}

}  // namespace tiny_llm_ext