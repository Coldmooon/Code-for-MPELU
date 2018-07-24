## Code-for-MPELU
Code for Improving Deep Neural Network with Multiple Parametric Exponential Linear Units, [arXiv:1606.00305](https://arxiv.org/abs/1606.00305)

The main contributions are:

- A generalization of ELU which encompasses ReLU, LReLU and PReLU.
- A weight initialization, named taylor initialization, for very deep networks using ELU/MPELU. If used with ReLU neworks, it reduces to MSRA filler.
- A deep MPELU residual architecture that is more effective than the original (Pre-)ResNet one.

## Deep MPELU residual architecture

MPELU nopre bottleneck architecture:

![img](torch/models/MPELU-NoPre-ResNet.jpg)

## Test error on CIFAR-10/100

MPELU is initialized with alpha = 0.25 or 1 and beta = 1. The learning rate multipliers of alpha and beta are 5. The weight decay multipliers of alpha and beta are 5 or 10. The results are reported as best(mean ± std).

MPELU nopre ResNet | depth | #params | CIFAR-10 | CIFAR-100
-------|:--------:|:--------:|:--------:|:--------:|
alpha = 1; beta = 1 | 164 | 1.696M | 4.58 (4.67 ± 0.06) | 21.35 (21.78 ± 0.33)
alpha = 1; beta = 1 | 1001 | 10.28M | 3.63 (3.78 ± 0.09) | 18.96 (19.08 ± 0.16)
alpha = 0.25; beta = 1 | 164 | 1.696M | 4.43 (4.53 ± 0.12) | 21.69 (21.88 ± 0.19)
alpha = 0.25; beta = 1 | 1001 | 10.28M | **3.57 (3.71 ± 0.11)** | **18.81 (18.98 ± 0.19)**

To replicate our results,

1. Install [fb.resnet.troch](https://github.com/facebook/fb.resnet.torch)
2. Follow our instructions to install MPELU in torch.
2. Copy files in `mpelu_nopre_resnet` to `fb.resnet.torch` and overwrite the original files.
3. Run the following command to train a 1001-layer MPELU nopre ResNet

```
th main.lua -netType mpelu-preactivation-nopre -depth 1001 -batchSize 64 -nGPU 2 -nThreads 12 -dataset cifar10 -nEpochs 300 -shortcutType B -shareGradInput false -optnet true | tee checkpoints/log.txt
```

## Installation
We provide [Caffe](https://github.com/BVLC/caffe) and [Torch](http://torch.ch/) implementations.

### Caffe:

1) Download the latest `caffe` from [https://github.com/BVLC/caffe](https://github.com/BVLC/caffe)

2) Move `caffe/*` of this repo to the `caffe` directory and follow the [instruction](http://caffe.berkeleyvision.org/installation.html) to compile.

### Torch:

1) Update `torch` to the latest version. This is necessary because of [#346](https://github.com/torch/cunn/pull/346).

2) Move `torch/extra` in this repo to the official torch directory and overwrite the corresponding files.

3) Run the following command to compile new layers.

```
cd torch/extra/nn/
luarocks make rocks/nn-scm-1.rockspec
cd torch/extra/cunn/
luarocks make rocks/cunn-scm-1.rockspec
```

## Usage
### Caffe:

**MPELU**:
In caffe, MPELU exists as the `M2PELU` layer, where `2` for `two parameters` alpha and beta which are both initialized as 1 in default.
To simply use this layer, replace `type: "ReLU"` with `type: "M2PELU"` in network defination files.

**Taylor filler**:
First, replace the keyword `gaussian` or `MSRA` with `taylor` in the `weight_filler` domain. Then, Add two new lines to specify values of `alpha` and `beta`:

```
weight_filler {
      type: "taylor"
      alpha: 1
      beta: 1
}
```
See the examples for details.


### Torch

I implemented two activation functions, `SPELU` and `MPELU`, where `SPELU` is a trimmed version of MPELU and can also be seen as a learnable `ELU`.

```
nn.SPELU(alpha=1, nOutputPlane=0)
nn.MPELU(alpha=1, beta=1, nOutputPlane=0)
```

- When `nOutputPlane = 0`, the `channel-shared` version will be used. 
- When `nOutputPlane` is set to the number of feature maps, the `channel-wise` version will be used.

To set the multipliers of weight decay for `MPELU`, use the `nnlr` package.

```
$ luarocks install nnlr
```

```
require 'nnlr'

nn.MPELU(alpha, beta, channels):learningRate('weight', lr_alpha):weightDecay('weight', wd_alpha)
                               :learningRate('bias', lr_beta):weightDecay('bias', wd_beta)
```

**Taylor filler**: Please check our examples in `mpelu_nopre_resnet`.








