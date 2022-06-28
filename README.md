# NNEquiv
NNEquiv is a neural network equivalence verification tool based on the geometric path enumeration algorithm.

This implementation is based on the [NNEnum](https://github.com/stanleybak/nnenum) tool by [Stanley Bak](http://stanleybak.com) for single neural networks.

## Getting started
To get started it's best to have a look at `examples/equiv/test.py` which explains how equivalence properties can be verified.
You can also invoke the approach by running:
```
examples/equiv/test.py [first-net] [second-net] [input space] [property] [strategy]
```
where:
- `[first-net]` and `[second-net]` are `ONNX` networks
- `[input space]` is an input space defined in `examples/equiv/properties.py`
- `[property]` is either `top` or the epsilon value to be proven (e.g. `0.05`)
- `[strategy]` is the refinement strategy (no refinement: `DONT`)

## Experimental Evaluation
If you are looking for further information on the experimental evaluation of this tool,
you might be interested in [this repository](https://github.com/samysweb/nnequiv-experiments)

## Citation
If you use this work in your research please:
- Cite the work by [Stanley Bak on which we base our implementation](https://link.springer.com/chapter/10.1007/978-3-030-53288-8_4)
- Cite [our work which extends the Geometric Path Enumeration Algorithm to multiple networks](https://ieeexplore.ieee.org/document/9643328)