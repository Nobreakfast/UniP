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
	- `torch2onnx`
    - `deform_conv2d_onnx_exporter`

## Install from source
``` shell
git clone https://github.con/Nobreakfast/UniP.git
cd UniP
checkout v2.0.0
pip install -e .
```

## Minimal Example

## Advanced Example

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

## Bugs
## Energy Calculation Part
### Requirements
- `pynvml`: 11.5.0
- `pyRAPL`: 0.2.3.1

### Example: calculation of whole model
``` python
import torch
import torchvision.models as models
from unip.utils.energy import Calculator

calculator = Calculator(device_id=6)

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

### `v2.0.0`: Branch new UniP Version

### `v1.0.0`: First Version of UniP

## Acknowledge
- [torch_pruning](https://github.com/VainF/Torch-Pruning)


