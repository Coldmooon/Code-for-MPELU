# Code-for-MPELU
Code for Improving Deep Neural Network with Multiple Parametric Exponential Linear Units, [arXiv:1606.00305](https://arxiv.org/abs/1606.00305)

This paper includes two contributions:
1) A generalization of ELU which encompasses ReLU, LReLU and PReLU.
2) A weight initialization, named taylor initialization, for ELU/MPELU which can also be used in ReLU and PReLU cases.

# Install
This implementation is based on [Caffe](https://github.com/Coldmooon/caffe) and compatible with the latest version.
Just copy files into caffe and follow the [instruction](http://caffe.berkeleyvision.org/installation.html) to compile.

# Usage
MPELU: 
In caffe, MPELU exists as the `M2PELU` layer, where `2` for `two parameters` alpha and beta which are both initialized as 1 in default.
To simply use this layer, replace `type: "ReLU"` with `type: "M2PELU"` in network defination files.

Taylor filler:
First, replace the keyword `gaussian` or `MSRA` with `taylor` in the `weight_filler` domain. Then, Add two new lines to specify values of alpha and beta.

```
weight_filler {
      type: "taylor"
      alpha: 1
      beta: 1
}
```

