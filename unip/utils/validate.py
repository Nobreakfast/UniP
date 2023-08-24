from unip.core.pruner import BasePruner


def validate(model, example_input, igtype2nodetype):
    BP = BasePruner(
        model,
        example_input,
        "RandomRatio",
        igtype2nodetype=igtype2nodetype,
        algo_args={"score_fn": "randn"},
    )
    BP.algorithm.run(0.6)
    BP.prune()
    if isinstance(example_input, (tuple, list)):
        model(*example_input)
    elif isinstance(example_input, dict):
        model(**example_input)
    else:
        model(example_input)
