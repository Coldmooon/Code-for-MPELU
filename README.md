## Code-for-MPELU
Code for Improving Deep Neural Network with Multiple Parametric Exponential Linear Units, [arXiv:1606.00305](https://arxiv.org/abs/1606.00305)

The main contributions are:

- A generalization of ELU which encompasses ReLU, LReLU and PReLU.
- A weight initialization, named taylor initialization, for very deep networks using ELU/MPELU. If used with ReLU neworks, it reduces to MSRA filler.
- A deep MPELU residual architecture that is more effective than the original (Pre-)ResNet one.

##Deep MPELU residual architecture

MPELU nopre bottleneck architecture:

![img](torch/models/MPELU-NoPre-ResNet.jpg)

##Test error on CIFAR-10/100

MPELU nopre ResNet: best(mean ± std)

Network | alpha & beta | depth | #params | CIFAR-10 | CIFAR-100 
-------|:-------:|:--------:|:--------:|:--------:|:--------:|
MPELU nopre ResNet |alpha = 1; beta = 1 | 164 | 1.696M | 4.58 (4.67 ± 0.06) | 21.35 (21.78 ± 0.33)
MPELU nopre ResNet |alpha = 1; beta = 1 | 1001 | 10.28M | 3.63 (3.78 ± 0.09) | 18.96 (19.08 ± 0.16)
MPELU nopre ResNet |alpha = 0.25; beta = 1 | 164 | 1.696M | 4.43 (4.53 ± 0.12) | 21.69 (21.88 ± 0.19)
MPELU nopre ResNet |alpha = 0.25; beta = 1 | 1001 | 10.28M | **3.57 (3.71 ± 0.11)** | **18.81 (18.98 ± 0.19)**


## Installation
We provide [Caffe](https://github.com/Coldmooon/caffe) and [Torch](http://torch.ch/) implementations.

###Caffe:

1) Download the latest caffe from [https://github.com/Coldmooon/caffe](https://github.com/Coldmooon/caffe)

2) Move `caffe/*` of this repo to the caffe directory you just downloaded and follow the [instruction](http://caffe.berkeleyvision.org/installation.html) to compile.

###Torch:

1) Update `torch` to the latest version. This is necessary because of [#346](https://github.com/torch/cunn/pull/346).

2) install nnlr.

```
$ luarocks install nnlr
```

3) Move `torch/extra` in this repo to the official torch directory and overwrite the corresponding files.


4) naive version 1: MPELU = PReLU(ELU(PReLU(x))). 

Just include `torch/naive-mpelu.lua` in your code. For example:

```
require '/path/to/naive-mpelu'
```
naive version 2: MPELU = **ELU'**(PReLU(x)),

where **ELU'** (implemented as `SPELU` in our code) means ELU with a learnbale parameter. Finally, compile the new layer `SPELU`:

```
cd torch/extra/nn/
luarocks make rocks/nn-scm-1.rockspec
cd torch/extra/cunn/
luarocks make rocks/cunn-scm-1.rockspec
```
To use `MPELU`, just include `/path/to/mpelu.lua` in your code.


## Usage
###Caffe:

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

###Torch

**MPELU of naive version**:

```
require '/path/to/naive-mpelu'

model = nn.Sequential()
model:add(MPELU(alpha, beta, alpha_lr_mult, beta_lr_mult, alpha_wd_mult, beta_wd_mult, num_of_channels))
```

**MPELU corresponding to Eqn.(1) in our paper**:


```
require '/path/to/mpelu'

model = nn.Sequential()
model:add(MPELU(alpha, beta, alpha_lr_mult, beta_lr_mult, alpha_wd_mult, beta_wd_mult, num_of_channels))
```
This version is slightly faster than the naive one.

**Taylor filler**: Please check our examples.









