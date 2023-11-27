import torch
import mpelu_cuda
from torch.cuda.amp import custom_fwd, custom_bwd

class MPELUFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, alpha, beta):
        output = mpelu_cuda.mpelu_forward(input, alpha.to(input.dtype), beta.to(input.dtype))
        ctx.save_for_backward(input, alpha, beta, output)

        return output


    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, alpha, beta, output = ctx.saved_tensors
        alpha = alpha.to(input.dtype)
        beta = beta.to(input.dtype)
        grad_input = torch.zeros_like(input)
        grad_a = torch.zeros_like(alpha)
        grad_b = torch.zeros_like(beta)

        mpelu_cuda.mpelu_backward(input, alpha, beta, output, grad_output.contiguous(), grad_input.contiguous(), grad_a.contiguous(), grad_b.contiguous())
        
        return grad_input, grad_a, grad_b


class MPELU(torch.nn.Module):
    def __init__(self, num_channels):
        super(MPELU, self).__init__()
        self.alpha = torch.nn.Parameter(torch.Tensor(num_channels))
        self.beta = torch.nn.Parameter(torch.Tensor(num_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.alpha, 0.25)
        torch.nn.init.ones_(self.beta)

    def forward(self, input):
        return MPELUFunction.apply(input, self.alpha, self.beta)