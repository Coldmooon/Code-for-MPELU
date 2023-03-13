#include "mpelu.h"

torch::Tensor mpelu_forward(
    torch::Tensor input,
    torch::Tensor a,
    torch::Tensor b
) {
    CHECK_INPUT(input);
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    const int channel = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    return mpelu_forward_cuda(input, a, b, channel, height, width);
}

void mpelu_backward(
    const torch::Tensor& input,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& grad_output,
    torch::Tensor& grad_input,
    torch::Tensor& grad_a,
    torch::Tensor& grad_b
) {
    CHECK_INPUT(input);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(grad_input);
    CHECK_INPUT(grad_a);
    CHECK_INPUT(grad_b);

    mpelu_backward_cuda(input, a, b, grad_output, grad_input, grad_a, grad_b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("mpelu_forward", &mpelu_forward);
    m.def("mpelu_backward", &mpelu_backward);
}

