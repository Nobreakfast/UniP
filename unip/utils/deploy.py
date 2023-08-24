import torch
import onnx
import onnxruntime
import onnx2torch as o2t
import numpy as np
import deform_conv2d_onnx_exporter

deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()


def output2name(output, name="output"):
    name_list = []
    if isinstance(output, (tuple, list)):
        for i, o in enumerate(output):
            name_list.extend(output2name(o, f"{name}_{i}"))
    else:
        name_list.append(name)
    return name_list


def torch2onnx(model, example_input, onnx_path):
    model.eval()
    if isinstance(example_input, (tuple, list)):
        input_names = ["input_{}".format(i) for i in range(len(example_input))]
        example_out = model(*example_input)
    elif isinstance(example_input, dict):
        input_names = list(example_input.keys())
        example_out = model(**example_input)
    else:
        input_names = ["input"]
        example_out = model(example_input)
    if isinstance(example_out, (tuple, list)):
        output_names = output2name(example_out)
    else:
        output_names = ["output"]

    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
    )
    verify_onnx(onnx_path)


def onnx2torch(onnx):
    return o2t.convert(onnx)


# TODO
def onnx2trt(onnx_model):
    pass


# TODO
def torch2trt(model, example_input):
    pass


def verify_onnx(onnx_path):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX check passed!")
    return onnx_model


def inference_onnx(onnx_path, example_input):
    sess = onnxruntime.InferenceSession(onnx_path)
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [i.name for i in sess.get_outputs()]
    input_dict = {}
    if isinstance(example_input, (tuple, list)):
        for in_name, in_tensor in zip(input_names, example_input):
            input_dict[in_name] = in_tensor
    elif isinstance(example_input, dict):
        input_dict = example_input
    else:
        input_dict[input_names[0]] = example_input

    output = sess.run(output_names, input_dict)
    return output
