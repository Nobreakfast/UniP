import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

import unip
from unip.utils.evaluation import cal_flops, cal_fps
import time
from tqdm import trange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pt2fps(model_path):
    example_input = [torch.randn(1, 3, 320, 320), torch.randn(1, 3, 320, 320)]
    model = torch.load(model_path, map_location="cpu")
    flops, params = cal_flops(model, example_input, "cpu")

    fps = cal_fps(model, example_input, device, times=1000, warmup=400)
    print(model_path, "time:", 1 / fps, "fps", fps)
    return [model_path, 1 / fps, fps, flops, params]


if __name__ == "__main__":
    # read args model_folder
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, default="./models")
    args = parser.parse_args()

    # open folder, read all models
    import os

    model_list = os.listdir(args.model_folder)
    model_list = [os.path.join(args.model_folder, model) for model in model_list]

    # test all models
    results = []
    for model_path in model_list:
        results.append(pt2fps(model_path))

    # save results to csv
    results = np.array(results)
    np.savetxt(
        "Achelous_prune_fps.csv",
        results,
        fmt="%s",
        delimiter=",",
        header="model_pt,time,fps,flops,params",
    )
