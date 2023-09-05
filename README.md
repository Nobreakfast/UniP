# UniP
A unified framework for Multi-Modality Pruning

## Requirements
- `torch`
- `torchvision`
- `numpy`
- `tqdm`
- `thop`
- (optional)
    - `pynvml`
    - `pyRAPL`
    - `jtop`
	- `torch2onnx`
    - `deform_conv2d_onnx_exporter`

## Install from source
``` shell
git clone https://github.con/Nobreakfast/UniP.git
cd UniP
pip install -e .
```

## Minimal Example
``` python
import torch
import torchvision.models as models
from unip.core.pruner import BasePruner
from unip.utils.evaluation import cal_flops

# load model and example input
model = models.resnet18()
example_input = torch.rand(1, 3, 224, 224, requires_grad=True)

# record the flops and params
cal_flops(model, example_input, device)

# define pruner
BP = BasePruner(
    model,
    example_input,
    "UniformRatio",
    algo_args={"score_fn": "weight_sum_l1_out"},
)
pruner.prune(0.3)

# record the flops and params
cal_flops(model, example_input, device)
```

## Advanced Example
``` python
import sys
sys.path.append("./tests")
import torch
from model.radarnet import RCNet
from model.backbone.conv_utils.dcn import DeformableConv2d
from unip.core.pruner import BasePruner
from unip.utils.evaluation import cal_flops
# RCNet need more customized node for deformable conv
from unip.core.node import dcnNode

# load model and example input
model = RCNet(in_channels=3)
example_input = torch.randn(1, 3, 320, 320, requires_grad=True)

# record the flops and params
cal_flops(model, example_input)

# define a dict indicate the module with node
igtype2nodetype = {DeformableConv2d: dcnNode}

# define pruner
BP = BasePruner(
    model,
    example_input,
    "UniformRatio",
    algo_args={"score_fn": "weight_sum_l1_out"}, 
    igtype2nodetype=igtype2nodetype,
)
BP.prune(0.3)

# record the flops and params
cal_flops(model, example_input, device)
```

## More examples
Please refer to the `./tutorials` folder for more examples.

## Troubleshooting

## Report a bug
Before report bugs by opening an issue in the [GitHub Issue Tracker](https://github.com/Nobreakfast/UniP/issues/new), you may follow the instructions of [test_Unip README.md](https://github.com/Nobreakfast/test_UniP). Basically, you need to provide the following information:
- the model you want to prune
- an example inference
Once you new a issue, we will fix it as soon as possible.

## Contributions
You are welcome to contribute to this project. Please follow the [Contribution Guide]().

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
    - classification:
        - [x] GoogleNet
        - [x] VGG
        - [x] ResNet, WideResNet, 
        - [x] MobileNetV2, MobileNetV3
        - [x] EfficientNet, EfficientNetV2
        - [x] SqueezeNet
    - [x] mobilevit
    - [x] achelous
    - [x] Edge_end_Multi_model_Visual_Grounding_Framework
- Tested modules:
    - conv: basic conv, depth-wise conv, deformable conv (with Custom Node `dcnNode`)
    - linear: basic linear, linear with input more than 2 dimensions
    - calculation operators: concat, split, slice, reshape, permute, expand, transpose, matmul, upsample, add, sub, div, mul, squeeze, unsqueeze
    - fired design: residual, ghost module, attention, shuffle attention
- support backwards type
    - [x] ConvolutionBackward0
    - [x] AddmmBackward0, MmBackward0, BmmBackward0
    - [x] AddBackward0, SubBackward0, MulBackward0, DivBackward0
    - [x] NativeBatchNormBackward0, NativeLayerNormBackward0, NativeGroupNormBackward0
    - [x] CatBackward0, SplitBackward0, SliceBackward0
    - [x] ReshapeAliasBackward0, UnsafeViewBackward0, ViewBackward0, PermuteBackward0, ExpandBackward0, TransposeBackward0, SqueezeBackward1, UnsqueezeBackward0, 
    - [x] MeanBackward1, MaxPool2DWithIndicesBackward0, AvgPool2DBackward0, UpsampleBilinear2DBackward0, UpsampleNearest2DBackward0, UpsampleBicubic2DBackward0
    - [x] ReluBackward0, SiluBackward0, GeluBackward0, HardswishBackward0, SigmoidBackward0, TanhBackward0, SoftmaxBackward0, LogSoftmaxBackward0
    - [x] AccumulateGrad, TBackward0, CloneBackward0
    - [x] RepeatBackward0
    - [x] EmbeddingBackward0
    - [ ] AsStridedBackward0
    - [ ] CopySlices
    - [ ] UnsafeSplitBackward0
    - [ ] UnbindBackward0
    - [ ] IndexBackward0
	- [ ] SqueezeBackward0
	- [ ] MaxBackward0
	- [ ] UnsafeSplitBackward0
	- [ ] StackBackward0
	- [ ] TransposeBackward1

## Bugs
- [ ] when `in_channels` greater than `groups`
- [ ] when operation `conv` and `fc` does not use `PyTorch` module implementation
- [x] fix the bug of useless `split`, when the prev_node's group is `non-prunable`
- [x] `TransposeConv` error
- [x] for some nodes starting from a non-`Input`, the dim_offset is wrong
- [x] `ConcatNode` is the next of `ReshapeNode`
- [x] `nn.MultiheadAttention` module not working: fixed by adding `CustomNode`
- [x] RCNet may failed cuz the residual with input
- [x] fix the bug for `DCN` module: use `dcnNode`
- [x] fix the bug for such module like GhostModule, use Non-`InOutNode` before `OutputNode`
- [x] does not prune the `LastLienarNode` for `to_qkv` like module
- [x] need to fix the `round_to` like attribute for `to_qkv` like module
- [x] `dim_offset` for reshape node is not always correct

## Energy Calculation Part
### Requirements
- `pynvml`: 11.5.0
- `pyRAPL`: 0.2.3.1

Note: require to give read permission to the specific file:
``` shell
sudo chmod -R a+r /sys/class/powercap/intel-rapl
```
### Example: calculation of whole model
``` python
import torch
import torchvision.models as models
from unip.utils.energy import Calculator

# define a device dict
device_dict = {
    "NvidiaDev": {'device_id': 0},
    "IntelDev": {},
    }
calculator = Calculator(device_dict)

model = models.resnet18()
model.eval().cuda()
example_input = torch.randn(1, 3, 224, 224).cuda()

@calculator.measure(times=1000)
def inference(model, example_input):
    model(example_input)

inference(model, example_input)
```

### Example: calculation of single module
``` python
import torch
import torchvision.models as models
from unip.utils.energy import Calculator

calculator = Calculator(cpu=False, device_id=4)

model = models.resnet18()
model.eval().cuda()

def forward_hook_record_input(module, input, output):
    setattr(module, "input", input[0])

hook = model.layer1.register_forward_hook(forward_hook_record_input)
example_input = torch.randn(1, 3, 224, 224).cuda()
model(example_input)
hook.remove()

@calculator.measure(times=100)
def inference(model, example_input):
    model(example_input)

inference(module, torch.randn_like(module.input))
```

## Change Log

### `v1.0.5`: 2023-09-01 Fix bugs for `v1.0.4`, add features, and optimize the project (on-going)
- new features:
    <!-- - add `GlobalAlgo` for global pruning -->
    - add better inheritance for `Multi-Modality Pruning`
    - add `github wiki` page for better documentation
    - add `tutorial/examples/*` for better usage
	<!-- - add support for `UnsafeSplitBackward0`, `UnbindBackward0`, `IndexBackward0`, `Squeezebackward0`, `Maxbackward0`, `UnsafeSplitBackward0`, `StackBackward0`, `TransposeBackward1` -->
- changes:
    - add examples in `./tutorial/examples`
    - optimize `energy.py` for jetson
    - optimize `evaluation.py` for fps
- bug fixing:
    - fix the bug of useless `split`, when the prev_node's group is `non-prunable`
### `v1.0.4`: 2023-08-25 Fix bugs for `v1.0.3`, add features, and optimize the project
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
### `v1.0.3`: 2023-08-21 Fix bugs for `v1.0.2` and optimize the project
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

## Acknowledge
- [torch_pruning](https://github.com/VainF/Torch-Pruning)


