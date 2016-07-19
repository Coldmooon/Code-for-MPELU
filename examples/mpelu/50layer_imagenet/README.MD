# Comparison of weight initialization

The 52-layer network initialized by taylor is able to converge without Batch Normalization.

Run the following command to check:

```
build/tools/caffe train -solver examples/mpelu/solver_elu.prototxt
```

As a counterpart, LSUV([arXiv:1511.06422](http://arxiv.org/abs/1511.06422)) can also be used to initialize ELU network. The code is available in [LSUVinit](https://github.com/ducha-aiki/LSUVinit).However, 
in our experiments, LSUV made the 52-layer network(also 30-layer ELU network) explode within several iterations.

To initialize ELU network with LSUV, first run the following command to compute weights:

```
python examples/mpelu/lsuv_init_python3.py examples/mpelu/layer52_elu_lsuv_withoutBN.prototxt elu.lsuv  OrthonormalLSUV
```
The computed lsuv weights will be saved in the file `elu.lsuv`.

Then, fine-tune the network from lsuv weights `elu.lsuv`.

```
build/tools/caffe train -solver examples/mpelu/solver_elu.prototxt -weights elu.lsuv
```