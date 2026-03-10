#pragma once

#include <mlx/array.h>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace tiny_llm_ext {

void load_library(mx::Device d, const char *path);
mx::array quantized_matmul(const mx::array &x,       // M × N (float16, activations)
                           const mx::array &weight,  // Input array K × (N/8) (uint32, packed weights)
                           const mx::array &scales, const mx::array &biases, int group_size, int bits, bool transpose_b,
                           mx::StreamOrDevice s = {}  // Stream on which to schedule the operation
);
class QuantizedMatmul : public mx::Primitive {
private:
    int group_size_;
    int bits_;

public:
    explicit QuantizedMatmul(mx::Stream stream, int group_size, int bits)
        : mx::Primitive(stream), group_size_(group_size), bits_(bits) {};
    void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
    /**
     * The primitive must know how to vectorize itself across
     * the given axes. The output is a pair containing the array
     * representing the vectorized computation and the axis which
     * corresponds to the output vectorized dimension.
     */
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &inputs,
                                                             const std::vector<int> &axes) override;

    /** Print the primitive. */
    void print(std::ostream &os);

    /** Name of the primitive (not virtual in some MLX versions). */
    const char *name() const override { return "QuantizedMatmul"; }

    /** Equivalence check **/
    bool is_equivalent(const mx::Primitive &other) const override;
};
}  // namespace tiny_llm_ext
