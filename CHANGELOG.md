
# Changelog
## `v1.0.6`: 2023-09-20 Fix bugs for `v1.0.5`, add features, and optimize the project (on-going)
- new features:
    - add `MMMTU` and `AdaptiveMMU`. And add a function to extract tags in `unip/core/node.py`
    <!-- - add `GlobalAlgo` for global pruning -->
	<!-- - add support for `UnsafeSplitBackward0`, `UnbindBackward0`, `IndexBackward0`, `Squeezebackward0`, `Maxbackward0`, `UnsafeSplitBackward0`, `StackBackward0`, `TransposeBackward1` -->
- changes:
    - split the `changelog` into `CHANGELOG.md`
- bug fixing:
    - fix the problem of installation guide
    <!-- - fix the problem of energy examples in `tests` -->
## `v1.0.5`: 2023-09-13 Fix bugs for `v1.0.4`, add features, and optimize the project
- new features:
    - now, we are available in [PyPi](https://pypi.org/project/Unified-Pruning/). `Note`: Please follow the installation guide in [README.md](https://github.com/Nobreakfast/UniP), because the description of [PyPi](https://pypi.org/project/Unified-Pruning/) is wrong, and will be fixed in next version.
    - add better inheritance for `Multi-Modality Pruning`
    - add `github wiki` page for better documentation
    - add `tutorial/examples/*` for better usage
- changes:
    - add examples in `./tutorial/examples`
    - optimize `energy.py` for jetson
    - optimize `evaluation.py` for fps
- bug fixing:
    - fix the bug of useless `split`, when the prev_node's group is `non-prunable`
## `v1.0.4`: 2023-08-25 Fix bugs for `v1.0.3`, add features, and optimize the project
- new features:
    - add support for `torch 1.xx.xx`
	- add `prune_transposeconv` to `prune_ops.py`
	- add `utils/validate.py` for some handy functions to test new model
	- add `RePeatBackward0` and `RepeatNode`
	- add `deploy.py`
- changes:
	- change the `fc` layer detection for `AddBackward0` in `core/pruner.py`
    - add `nn.Dropout` layer detection, and pass the grad_fn
    - optimize the `prune()` function for `InoutNode` 
    - optimize the `in_shape` and `out_shape` of `ConcatNode`
    - optimize and add `split` attribute to `ReshapeNode` for `channel shuffle`
	- optimize the `RatioAlgo`, and fixed the bug.
- bug fixing:
    - fix the bug for `ConcatNode` is the next of `ReshapeNode`
    - `nn.MultiheadAttention` module not working, add `CustomNode` for it
    - for some nodes starting from a non-`Input`, the dim_offset is wrong: we ignore the nodes
	- `TransposeConv` error
## `v1.0.3`: 2023-08-21 Fix bugs for `v1.0.2` and optimize the project
- new features:
    - add `energy` calculation for `NVIDIA` GPU and `Intel` CPU:
      Note: require to give read permission to the specific file:
      ``` shell
      sudo chmod -R a+r /sys/class/powercap/intel-rapl
      ```
      This may has some problem, as it could not record the simutaneous energy consumption. Besides, it need to be changed to multi-thread version.
    - add example code for `energy` calculation in `README.MD`
    - add model saving and loading methods
    - add `weight_sum_l1_out` score function
    - add `name2node`, `name2score`, and `name2algo`
      
- changes:
    - organize the `BaseAlgo` for better inheritance
        - add `RatioAlgo` and `GlobalAlgo` as the lower level of `BaseAlgo`
        - change `UniformAlgo` and `RandomAlgo` as the lower level of `RatioAlgo`
        - add `param` attribute `Node` for Algorithm score function
        - rename the Third-level class of `BaseAlgo` to `UnifomRatio` and `RandomRatio`
    - optimize the node for better saving and loading strategy
    - organize the `./ttl` folder for better example
        - change `./ttl` to `./tutorials`
        - some key actions are added to the `./tutorials/utils` folder
        - the `./tutorials/examples` folder will be used for the example usage of UniP
    - organize the `./test` folder for better test
- bug fixing:
    - fix the bug of RCNet: may failed cuz the residual with input
## `v1.0.2`: 2023-08-14 Fix bugs for `v1.0.1` and add features
- new features:
    - add `ignore_list` for some unwanted and unsupported modules
    - add `CustomNode` for custom module
    - add support for `SliceBackward0`
    - add support for `SqueezeBackward1` and `UnsqueezeBackward0`
    - add support for `UpsampleBilinear2DBackward0`, `UpsampleNearest2DBackward0`, `UpsampleBicubic2DBackward0`
    - add support for `GhostModule` as the `Slice` node is supported
- changes:
    - remove the `hasdummy`, and search the prev_node of `DummyNode` in `BaseGroup` instead
    - change `update_dim_offset` method
- bug fixing:
    - fix the bug for `DCN` module: use `dcnNode`
    - fix the bug for such module like GhostModule, use Non-`InOutNode` before `OutputNode`
## `v1.0.1`: 2023-08-13 Fix bugs for `v1.0.0` and add features
- new features:
    - new dim change calculation method as patches for `ReshapeNode`
- changes:
    - move `MatmulNode` to `ChangeNode` from `InInNode` as its previous node could not be in the same group
- bug fixing:
    - does not prune the `LastLienarNode` for `to_qkv` like module
    - need to fix the `round_to` like attribute for `to_qkv` like module
    - `dim_offset` for reshape node is not always correct
## `v1.0.0`: 2023-08-10 Finish the basic framework of UniP 
- Add nodes definition: `BaseNode`, `InOutNode`, `OutOutNode`, `InInNode`, `RemapNode`, `ChangeNode`, `DummyNode`, `ActivationNode`, `PoolNode` and `CustomNode`
- Add groups definition: `BaseGroup`, `CurrentGroup`, and `NextGroup`
- Add algorithm definition: `BaseAlgo`, `RandomAlgo`, and `UniformAlgo`
- Add pruner
- Add three example models to test the framework
- Add module pruning functions: conv, linear, bn, and bundle parameters

