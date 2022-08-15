#ifndef _KDKNN_JAX_KERNELS_H_
#define _KDKNN_JAX_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace kdknn_jax {
struct KdknnDescriptor {
  std::int64_t size;
};

void gpu_kdknn_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);
void gpu_kdknn_f64(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

}  // namespace kdknn_jax

#endif