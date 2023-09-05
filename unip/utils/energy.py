import time
import functools
import threading
import abc

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


def __get_dev(dev_name):
    return globals()[dev_name]


class BaseDev(abc.ABC):
    def __init__(self, name):
        self.name = name
        self.init_time = time.time()
        self.init_power = self.get_time_power()[1]
        self.zero_energy()
        self.stop_flag = False

    def thread_function(self):
        print(f"Thread {self.name}: starting")
        while True:
            self.step()
            time.sleep(0.001)
            if self.stop_flag:
                break
        print(f"Thread {self.name}: finishing")

    def start(self):
        self.thread = threading.Thread(target=self.thread_function)
        self.thread.start()

    def end(self):
        self.stop_flag = True
        time.sleep(0.002)

    def zero_energy(self):
        self.power_list = [self.get_time_power()]
        self.energy = 0
        self.stop_flag = False

    def step(self):
        [time_n, power] = self.get_time_power()
        power_interval = ((power + self.power_list[-1][1]) / 2) - self.power_list[0][1]
        time_interval = time_n - self.power_list[-1][0]
        self.energy += power_interval * time_interval
        self.power_list.append([time_n, power])

    def summary(self, start_time, end_time, verbose=False):
        power_list = np.asarray(self.power_list)
        power_list = power_list[power_list[:, 0] > start_time]
        power_list = power_list[power_list[:, 0] < end_time]
        if len(power_list) < 2:
            return 0, 0
        power1 = power_list[:-1, 1] - self.init_power
        power2 = power_list[1:, 1] - self.init_power
        time = power_list[1:, 0] - power_list[:-1, 0]
        energy_all = (0.5 * (power1 + power2) * time).sum()
        power_all = energy_all / (power_list[-1, 0] - power_list[0, 0])
        if verbose:
            print("=" * 10, start_time, "~", end_time, "=" * 10)
            print(f"{self.name} Power: {power_all/1e3:3.5f} W")
            print(f"{self.name} Energy: {energy_all/1e3:3.5f} W*s")
        return power_all, energy_all

    @abc.abstractmethod
    def get_time_power(self):
        pass


class NvidiaDev(BaseDev):
    def __init__(self, device_id=0):
        import pynvml

        pynvml.nvmlInit()
        self.device_id = device_id
        self.handler = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
        super().__init__("Nvidia GPU")

    def get_time_power(self):
        time_n = time.time()
        power = pynvml.nvmlDeviceGetPowerUsage(self.handler)
        return [time_n, power]


class IntelDev(BaseDev):
    def __init__(self):
        import pyRAPL

        self.handler = pyRAPL.sensor.Sensor()
        super().__init__("Intel CPU")

    def get_time_power(self):
        time_n = time.time()
        power = np.asarray(self.handler.energy()).sum() / 1e6
        return [time_n, power]


class JetsonDev(BaseDev):
    def __init__(self, cpu=False):
        from jtop import jtop

        self.cpu = cpu
        self.handler = jtop()
        super().__init__("Jetson")

    def get_time_power(self):
        time_n = time.time()
        power = 0
        if self.cpu:
            power += self.get_cpu_power()
        if self.gpu:
            power += self.get_gpu_power()
        power = power / 1e6
        return [time_n, power]

    # TODO: Jetson
    def get_cpu_power(self):
        return 0

    # TODO: Jetson
    def get_gpu_power(self):
        return 0


def get_devices(dev_dict):
    devices = []
    for k, v in dev_dict.items():
        devices.append(__get_dev(k)(**v))
    return devices


class Calculator:
    def __init__(self, device: dict):
        self.devices = get_devices(device)

    def start(self):
        for dev in self.devices:
            dev.start()

    def end(self):
        for dev in self.devices:
            dev.end()

    def zero_energy(self):
        for dev in self.devices:
            dev.zero_energy()

    def summary(self, verbose=False):
        energy_all = 0
        power_all = 0
        devices_power = {}
        devices_energy = {}
        for dev in self.devices:
            power, energy = dev.summary(self.start_time, self.end_time, verbose)
            devices_power[dev.name] = power
            devices_energy[dev.name] = energy
            energy_all += energy
        power_all = energy_all / (self.end_time - self.start_time)
        if verbose:
            print(f"Power for All: {power_all/1e3:3.5f} W")
            print(f"Energy for All: {energy_all/1e3:3.5f} W*s")
        return power_all, energy_all, devices_power, devices_energy

    def summary_from_time(self, start_time, end_time, verbose=False):
        energy_all = 0
        power_all = 0
        devices_power = {}
        devices_energy = {}
        for dev in self.devices:
            power, energy = dev.summary(start_time, end_time, verbose)
            devices_power[dev.name] = power
            devices_energy[dev.name] = energy
            energy_all += energy
        power_all = energy_all / (end_time - start_time)
        if verbose:
            print(f"Power for All: {power_all/1e3:3.5f} W")
            print(f"Energy for All: {energy_all/1e3:3.5f} W*s")
        return power_all, energy_all, devices_power, devices_energy

    def measure(self, times=1000, warmup=0):
        @torch.no_grad()
        def _wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.zero_energy()
                self.start()
                if warmup != 0:
                    for i in trange(warmup, desc="Warmup", leave=False):
                        func(*args, **kwargs)
                self.start_time = time.time()
                for i in trange(times, desc="Measure", leave=False):
                    func(*args, **kwargs)
                self.end_time = time.time()
                self.end()
                self.summary(verbose=True)
                self.fps = times / (self.end_time - self.start_time)

            return wrapper

        return _wrapper


def forward_hook(module, input, output):
    if hasattr(module, "end_time"):
        module.end_time.append(time.time())
    else:
        setattr(module, "end_time", [time.time()])


def forward_pre_hook(module, input):
    if hasattr(module, "start_time"):
        module.start_time.append(time.time())
    else:
        setattr(module, "start_time", [time.time()])
