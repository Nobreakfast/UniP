import logging
from unip.core.pruner import name2pruner


def prune(pruner, model, example_data, verbose=False, **kwargs):
    if verbose:
        logging.basicConfig(level=logging.INFO)
    return name2pruner(pruner)(model, example_data, **kwargs)
