import numpy as np
import torch
import torch.nn as nn

import abc

from .nodes import *


def name2nodetype(name):
    return globals()[name]


