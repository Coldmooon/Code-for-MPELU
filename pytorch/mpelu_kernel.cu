#include <torch/extension.h>
#include <ATen/cuda/Atomic.cuh>

template <typename scalar_t>
__global__ void mpelu_forward_cuda_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> a,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> b,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output
    )
{
    int batch_size = input.size(0);
    int num_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int n = 0; n < batch_size; ++n) {
            for (int c = 0; c < num_channels; ++c) {
                scalar_t in_val = input[n][c][y][x];
                if (in_val < 0) {
                    scalar_t a_val = a[c];
                    scalar_t b_val = b[c];
                    output[n][c][y][x] = a_val * (exp(b_val * in_val) - 1);
                } else {
                    output[n][c][y][x] = in_val;
                }
            }
        }
    }

};

/*
=============== Solution 1 for atomicAdd not support for (c10::Half *, c10::Half) ===============
adapted from https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh
https://forums.developer.nvidia.com/t/atomicadd-not-overloaded-for-c10-half/204474/2
__device__ __forceinline__ void atomicAdd(c10::Half* address, c10::Half val) {
    unsigned int *address_as_ui = reinterpret_cast<unsigned int *>(reinterpret_cast<char *>(address) - (reinterpret_cast<size_t>(address) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        unsigned short hsum = reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
        hsum += val;
        old = reinterpret_cast<size_t>(address) & 2
                 ? (old & 0xffff) | (hsum << 16)
                 : (old & 0xffff0000) | hsum;
        old = atomicCAS(address_as_ui, assumed, old);

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
}
=============== End: Solution 1 for atomicAdd not support for (c10::Half *, c10::Half) ===============

=============== Solution 2 for atomicAdd not support for (c10::Half *, c10::Half) ===============
https://discuss.pytorch.org/t/c10-half-float-type-support-for-atomicadd/137628/2
Use gpuAtomicAdd rather than atomicAdd:
https://github.com/pytorch/pytorch/blob/085e2f7bddc45f859fcdb786926d60d709b2daa0/aten/src/ATen/cuda/Atomic.cuh#L181-L190
=============== End: Solution 2 for atomicAdd not support for (c10::Half *, c10::Half) =============== 
*/

template <typename scalar_t>
__global__ void mpelu_backward_cuda_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> a,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> b,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grad_output,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grad_input,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> grad_a,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> grad_b)
{
    
    const int64_t batch_size = grad_output.size(0);
    const int64_t num_channels = grad_output.size(1);
    const int64_t height = grad_output.size(2);
    const int64_t width = grad_output.size(3);
    const int64_t num_elements = batch_size * num_channels * height * width;


    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = index; i < num_elements; i += stride) {
        const int64_t n = (i / (num_channels * height * width)) % batch_size;
        const int64_t c = (i / (height * width)) % num_channels;
        const int64_t h = (i / width) % height;
        const int64_t w = i % width;

        const scalar_t inp = input[n][c][h][w];
        const scalar_t grad_out = grad_output[n][c][h][w];
        const scalar_t a_val = a[c];
        const scalar_t b_val = b[c];

        scalar_t grad_inp;
        if (inp >= 0) {
            grad_inp = grad_out;
        } else {
            grad_inp = grad_out * a_val * b_val * expf(b_val * inp);
            gpuAtomicAdd(&grad_a[c], grad_out * (expf(b_val * inp) - 1));
            gpuAtomicAdd(&grad_b[c], grad_out * a_val * inp * expf(b_val * inp));
        }

        grad_input[n][c][h][w] = grad_inp;
    }
}


// ===================================================================

torch::Tensor mpelu_forward_cuda(
    const torch::Tensor input,
    const torch::Tensor a,
    const torch::Tensor b,
    const int channel,
    const int height,
    const int width
){

    torch::Tensor output = torch::zeros_like(input);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (input.size(3) + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (input.size(2) + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "mpelu_forward_cuda", ([&] {
        mpelu_forward_cuda_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            a.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            b.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>()
            );
    }));
    
    return output;
}

void mpelu_backward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& a,
    const torch::Tensor& b,    
    const torch::Tensor& grad_output,
    torch::Tensor& grad_input,
    torch::Tensor& grad_a,
    torch::Tensor& grad_b
){

    grad_input.zero_();
    grad_a.zero_();
    grad_b.zero_();

    const int batch_size = grad_output.size(0);
    const int num_channels = grad_output.size(1);
    const int height = grad_output.size(2);
    const int width = grad_output.size(3);

    const int num_threads = 1024;
    const int num_blocks = (batch_size*num_channels*height*width + num_threads - 1) / num_threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.type(), "mpelu_backward_cuda", ([&] {
        mpelu_backward_cuda_kernel<scalar_t><<<num_blocks, num_threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            a.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            b.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            grad_output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            grad_input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            grad_a.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            grad_b.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>());
    }));
    
};