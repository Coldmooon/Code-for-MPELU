import torch
import mpelu_cuda
from torch.cuda.amp import autocast, custom_fwd, custom_bwd

class MPELUFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, a, b):
        with autocast():
            output = mpelu_cuda.mpelu_forward(input, a, b)
            ctx.save_for_backward(input, a, b)

            return output


    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, a, b = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_a = torch.zeros_like(a)
        grad_b = torch.zeros_like(b)
        mpelu_cuda.mpelu_backward(input, a, b, grad_output.contiguous(), grad_input.contiguous(), grad_a.contiguous(), grad_b.contiguous())

        return grad_input, grad_a, grad_b



class MPELU(torch.nn.Module):
    def __init__(self, num_channels):
        super(MPELU, self).__init__()
        self.a = torch.nn.Parameter(torch.Tensor(num_channels))
        self.b = torch.nn.Parameter(torch.Tensor(num_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.a, 0.25)
        torch.nn.init.ones_(self.b)

    def forward(self, input):
        return MPELUFunction.apply(input, self.a, self.b)
