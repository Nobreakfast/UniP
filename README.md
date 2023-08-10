# UniP
A unified framework for Pruning in Pytorch

## Support List
- Nodes for operations:
    - `BaseNode`: Base class for all other nodes
    - `InOutNode`: Node for conv, linear, and bundle parameters
    - `OutOutNode`: Node for bn, ln, gn, and dwconv
    - `InInNode`: Node for add, sub, mul, div, and matmul
    - `RemapNode`: Node for concat, and split
    - `ChangeNode`: Node for reshape, permute, expand, and transpose
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
- Tested modules:
    - [ ] conv
        - [x] basic conv
        - [x] depth-wise conv
        - [ ] deformable conv
    - [ ] linear
        - [x] basic linear
        - [ ] linear with input more than 2 dimensions
    - [ ] dim/shape change
        - [x] concat
        - [x] split
        - [x] reshape
        - [x] permute
        - [x] expand
        - [x] transpose
    - [ ] fired design
        - [x] residual
        - [x] ghost module
        - [x] attention
        - [x] shuffle attention

## Change Log
- 2023-08-10: Finish the basic framework of UniP (v1.0.0)
    - Add nodes definition: `BaseNode`, `InOutNode`, `OutOutNode`, `InInNode`, `RemapNode`, `ChangeNode`, `DummyNode`, `ActivationNode`, `PoolNode` and `CustomNode`
    - Add groups definition: `BaseGroup`, `CurrentGroup`, and `NextGroup`
    - Add algorithm definition: `BaseAlgo`, `RandomAlgo`, and `UniformAlgo`
    - Add pruner
    - Add three example models to test the framework
    - Add module pruning functions: conv, linear, bn, and bundle parameters