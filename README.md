# UniP
A unified framework for Pruning in Pytorch

## Support List
- Nodes for operations:
    - `BaseNode`: Base class for all other nodes
    - `InOutNode`: Node for conv, linear, and bundle parameters
    - `OutOutNode`: Node for bn, ln, gn, and dwconv
    - `InInNode`: Node for add, sub, mul, and div
    - `RemapNode`: Node for concat, split, and slice
    - `ChangeNode`: Node for reshape, permute, expand, transpose, matmul, and upsample
    - `DummyNode`: Node for dummy input and output
    - `ActivationNode`: Node for activation
    - `PoolNode`: Node for adaptive pooling, max pooling, and average pooling
    - `CustomNode`: Node for custom module
- Algorithm:
    - `BaseAlgo`: Base class for all other algorithms
    - `RandomAlgo`: Random ratio and index
    - `UniformAlgo`: Uniform ratio and random index
- Tested models:
    - [x] example model
    - [x] mobilevit
    - [x] achelous
- Tested modules:
    - conv
        - [x] basic conv
        - [x] depth-wise conv
        - [x] deformable conv (with Custom Node `dcnNode`)
    - linear
        - [x] basic linear
        - [x] linear with input more than 2 dimensions
    - calculation operators
        - [x] concat, split, slice
        - [x] reshape, permute, expand, transpose, matmul, upsample
        - [x] add, sub, div, mul
        - [x] squeeze, unsqueeze
    - fired design
        - [x] residual
        - [x] ghost module
        - [x] attention
        - [x] shuffle attention
    - support backwards type
        - [x] ConvolutionBackward0
        - [x] AddmmBackward0, MmBackward0, BmmBackward0
        - [x] AddBackward0, SubBackward0, MulBackward0, DivBackward0
        - [x] NativeBatchNormBackward0, NativeLayerNormBackward0, NativeGroupNormBackward0
        - [x] CatBackward0, SplitBackward0
        - [x] ReshapeAliasBackward0, UnsafeViewBackward0, ViewBackward0, PermuteBackward0, ExpandBackward0, TransposeBackward0, 
        - [x] MeanBackward1, MaxPool2DWithIndicesBackward0, AvgPool2DBackward0
        - [x] ReluBackward0, SiluBackward0, GeluBackward0, HardswishBackward0, SigmoidBackward0, TanhBackward0, SoftmaxBackward0, LogSoftmaxBackward0
        - [x] AccumulateGrad, TBackward0, CloneBackward0
        - [x] SliceBackward0
        - [x] SqueezeBackward1, UnsqueezeBackward0
        - [x] UpsampleBilinear2DBackward0,

## Bugs
- [x] fix the bug for `DCN` module: use `dcnNode`
- [x] fix the bug for such module like GhostModule, use Non-`InOutNode` before `OutputNode`
- [x] does not prune the `LastLienarNode` for `to_qkv` like module
- [x] need to fix the `round_to` like attribute for `to_qkv` like module
- [x] `dim_offset` for reshape node is not always correct

## Change Log
### `v1.0.3`: 2023-08-xx Fix bugs for `v1.0.2` and add features (ongoing)
- new features:
    <!-- - organize the `BaseAlgo` for better inheritance -->
    <!-- - organize the `./ttl` folder for better example -->
    <!-- - organize the `./test` folder for better test -->
- changes:
- bug fixing:
### `v1.0.2`: 2023-08-14 Fix bugs for `v1.0.1` and add features
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
### `v1.0.1`: 2023-08-13 Fix bugs for `v1.0.0` and add features
- new features:
    - new dim change calculation method as patches for `ReshapeNode`
- changes:
    - move `MatmulNode` to `ChangeNode` from `InInNode` as its previous node could not be in the same group
- bug fixing:
    - does not prune the `LastLienarNode` for `to_qkv` like module
    - need to fix the `round_to` like attribute for `to_qkv` like module
    - `dim_offset` for reshape node is not always correct
### `v1.0.0`: 2023-08-10 Finish the basic framework of UniP 
- Add nodes definition: `BaseNode`, `InOutNode`, `OutOutNode`, `InInNode`, `RemapNode`, `ChangeNode`, `DummyNode`, `ActivationNode`, `PoolNode` and `CustomNode`
- Add groups definition: `BaseGroup`, `CurrentGroup`, and `NextGroup`
- Add algorithm definition: `BaseAlgo`, `RandomAlgo`, and `UniformAlgo`
- Add pruner
- Add three example models to test the framework
- Add module pruning functions: conv, linear, bn, and bundle parameters