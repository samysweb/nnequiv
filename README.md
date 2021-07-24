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

## Further Remarks
Our work on equivalence verification will hopefully be published in a conference paper at some point in the near future.
If you want to cite this work in the meantime, please feel free to get in touch.
