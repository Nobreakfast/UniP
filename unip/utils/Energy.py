import time
import functools
import threading
import abc

import torch
import pynvml
import pyRAPL
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


class BaseCalculator(abc.ABC):
    def __init__(self, name):
        self.name = name
        self.init_time = time.time()
        self.zero_energy()
        self.stop_flag = False

    def thread_function(self):
        print(f"Thread {self.name}: starting")
        while True:
            self.step()
            time.sleep(0.01)
            if self.stop_flag:
                break
        print(f"Thread {self.name}: finishing")

    def start(self):
        self.thread = threading.Thread(target=self.thread_function)
        self.thread.start()

    def end(self):
        self.stop_flag = True
        time.sleep(0.02)
        # self.summary()

    def zero_energy(self):
        self.power_list = [self.get_time_power()]
        self.energy = 0
        # self.init_time = self.power_list[0][0]
        # self.init_power = self.power_list[0][1]
        self.stop_flag = False

    def step(self):
        [time_n, power] = self.get_time_power()
        power_interval = ((power + self.power_list[-1][1]) / 2) - self.power_list[0][1]
        time_interval = time_n - self.power_list[-1][0]
        self.energy += power_interval * time_interval
        self.power_list.append([time_n, power])

    def summary(self):
        # averge_power = self.energy / (self.power_list[-1][0] - self.power_list[0][0])
        # print(f"Average Power for {self.name}: {averge_power/1e3:3.5f} W")
        # print(f"Energy Costs for {self.name}: {self.energy/1e3:3.5f} W*s")
        # print(f"Energy Costs for {self.name}: {self.energy/1e6/3.6e3} kW*h")
        return self.summary_from_time(self.start_time, self.end_time)

    def summary_from_time(self, start_time, end_time):
        power_list = np.asarray(self.power_list)
        init_power = power_list[0, 1]
        power_list = power_list[power_list[:, 0] > start_time]
        power_list = power_list[power_list[:, 0] < end_time]
        power1 = power_list[:-1, 1] - init_power
        power2 = power_list[1:, 1] - init_power
        time = power_list[1:, 0] - power_list[:-1, 0]
        energy_all = (0.5 * (power1 + power2) * time).sum()
        power_all = energy_all / (power_list[-1, 0] - power_list[0, 0])
        print("=" * 10, start_time, "~", end_time, "=" * 10)
        print(f"{self.name} Power: {power_all/1e3:3.5f} W")
        print(f"{self.name} Energy: {energy_all/1e3:3.5f} W*s")
        return power_all, energy_all

    @abc.abstractmethod
    def get_time_power(self):
        pass


class GPUCalculator(BaseCalculator):
    def __init__(self, device_id=0):
        pynvml.nvmlInit()
        self.device_id = device_id
        self.handler = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
        super().__init__("GPU")

    def get_time_power(self):
        time_n = time.time() - self.init_time
        power = pynvml.nvmlDeviceGetPowerUsage(self.handler)
        return [time_n, power]


class CPUCalculator(BaseCalculator):
    def __init__(self):
        self.handler = pyRAPL.sensor.Sensor()
        super().__init__("CPU")

    def get_time_power(self):
        time_n = time.time() - self.init_time
        power = np.asarray(self.handler.energy()).sum() / 1e6
        return [time_n, power]


class CalculatorAPI:
    def __init__(self, device_id=0):
        self.GPU = GPUCalculator(device_id)
        self.CPU = CPUCalculator()

    def start(self):
        self.GPU.start()
        self.CPU.start()

    def end(self):
        self.GPU.end()
        self.CPU.end()

    def zero_energy(self):
        self.GPU.zero_energy()
        self.CPU.zero_energy()

    def summary(self):
        gpu_power, gpu_energy = self.GPU.summary()
        cpu_power, cpu_energy = self.CPU.summary()
        energy_all = gpu_energy + cpu_energy
        power_all = energy_all / (self.GPU.end_time - self.GPU.start_time)
        print(f"Power for All: {power_all/1e3:3.5f} W")
        print(f"Energy for All: {energy_all/1e3:3.5f} W*s")

    def summary_from_time(self, start_time, end_time):
        gpu_power = self.GPU.summary_from_time(start_time, end_time)
        cpu_power = self.CPU.summary_from_time(start_time, end_time)
        energy_all = gpu_power + cpu_power
        power_all = energy_all / (end_time - start_time)
        print(f"Power for All: {power_all/1e3:3.5f} W")
        print(f"Energy for All: {energy_all/1e3:3.5f} W*s")

    def measure(self, times=1000, warmup=0):
        @torch.no_grad()
        def _wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.zero_energy()
                self.start()
                if warmup != 0:
                    for i in trange(warmup):
                        func(*args, **kwargs)
                self.start_time = time.start()
                for i in trange(times):
                    func(*args, **kwargs)
                self.end_time = time.end()
                self.end()
                self.summary()

            return wrapper

        return _wrapper


def forward_hook(self, module, input, output):
    if hasattr(module, "end_time"):
        module.end_time.append(time.time())
    else:
        setattr(module, "end_time", [time.time()])


def forward_pre_hook(self, module, input):
    if hasattr(module, "start_time"):
        module.start_time.append(time.time())
    else:
        setattr(module, "start_time", [time.time()])


def example_model():
    import torch
    import torchvision.models as models

    model = models.resnet18()
    model.eval()
    model.cuda()
    example_input = torch.rand(1, 3, 224, 224).cuda()
    calculator = Calculator()

    @calculator.measure(times=1000)
    def inference(model, example_input):
        model(example_input)

    inference(model, example_input)


def example_module(times=100):
    import torch
    import torchvision.models as models

    model = models.resnet18()
    model.eval()
    model.cuda()
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(forward_hook))
        hooks.append(module.register_forward_pre_hook(forward_pre_hook))

    example_input = torch.rand(1, 3, 224, 224).cuda()
    calculator = Calculator()
    calculator.start()
    for i in trange(times):
        model(example_input)
    calculator.end()
    for hook in hooks:
        hook.remove()

    for name, module in model.named_modules():
        power_list, energy_list = [], []
        for i in range(len(module.start_time)):
            tmp_power, tmp_energy = calculator.summary_from_time(
                module.start_time[i], module.end_time[i]
            )
            power_list.append(tmp_power)
            energy_list.append(tmp_energy)
        power = np.asarray(power_list).sum() / times
        energy = np.asarray(energy_list).sum() / times
        print(f"{name} Power: {power/1e3:3.5f} W, Energy: {energy/1e3:3.5f} W*s")


if __name__ == "__main__":
    example_model()
    example_module()
