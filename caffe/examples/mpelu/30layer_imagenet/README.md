# Comparison of Convergence among activation functions

30-layer network without BatchNorm

|   Mode  | ReLU | PReLU | ELU | MPELU |
| --------|:----:|:-----:|:---:|:-----:|
| AVERAGE |  X   |   X   |  Y  |   Y   |
| FAN_IN  |  X   |   X   |  Y  |   Y   |
| FAN_OUT |  X   |   Y   |  Y  |   Y   |


- X: fail to converge
- Y: converge