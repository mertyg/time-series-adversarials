from .MLP import DummyNet, MLP
from .FCN import FCN

__model__ = {"Dummy": DummyNet,
             "MLP": MLP,
             "FCN": FCN}

def get_model(args):
    return __model__[args.model]
