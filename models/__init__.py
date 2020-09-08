from .MLP import DummyNet, MLP
from .FCN import FCN
from .resnet import TSResNet
from .shapelet import ShapeletNet

__model__ = {"Dummy": DummyNet,
             "MLP": MLP,
             "FCN": FCN,
             "resnet": TSResNet,
             "ShapeletNet": ShapeletNet}

def get_model(args):
    return __model__[args.model]
