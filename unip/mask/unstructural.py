import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class UnstructuredIndice(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        if self.indices != None:
            mask[self.indices] = 0
        return mask

    @classmethod
    def apply(cls, module, name, indices):
        return super(UnstructuredIndice, cls).apply(module, name, indices=indices)


def apply_prune(model, score_dict, threshold):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            indices = score_dict[name] < threshold
            UnstructuredIndice.apply(module, "weight", indices)


def cal_threshold(score_dict, ratio):
    all_scores = torch.cat([torch.flatten(x) for x in score_dict.values()])
    threshold = torch.kthvalue(all_scores, int(len(all_scores) * (ratio)))[0]
    return threshold


def cal_sparsity(model):
    num_zeros = 0
    num_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            num_zeros += module.weight.data.numel() - module.weight.data.nonzero().size(
                0
            )
            num_elements += module.weight.data.numel()
    return num_zeros / num_elements


def get_lw_sparsity(model):
    """
    get layer-wise sparsity
    """
    sparsity_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            num_zeros = module.weight.data.numel() - module.weight.data.nonzero().size(
                0
            )
            num_elements = module.weight.data.numel()
            sparsity_dict[name] = num_zeros / num_elements
    return sparsity_dict


def remove_mask(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.remove(module, "weight")


def get_mask(model):
    mask_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            mask_dict[name] = module.weight_mask
    return mask_dict


def get_weight(model):
    weight_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight_dict[name] = module.weight.data
    return weight_dict


def invert_score(score_dict):
    for key in score_dict.keys():
        score_dict[key] = -score_dict[key]
    return score_dict
