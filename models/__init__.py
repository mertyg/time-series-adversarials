from .MLP import DummyNet, MLP
from .FCN import FCN
from .resnet import TSResNet

__model__ = {"Dummy": DummyNet,
             "MLP": MLP,
             "FCN": FCN,
             "resnet": TSResNet}

def get_model(args):
    return __model__[args.model]
