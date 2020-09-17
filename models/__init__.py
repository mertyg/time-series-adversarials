from .MLP import DummyNet, MLP
from .FCN import FCN
from .resnet import TSResNet
from .shapelet import ShapeletNet
from .conv_shapelet import ConvShapeletNet

__model__ = {"Dummy": DummyNet,
             "MLP": MLP,
             "FCN": FCN,
             "resnet": TSResNet,
             "ShapeletNet": ShapeletNet,
             "ConvShapelet": ConvShapeletNet}


def get_model(args):
    return __model__[args.model]
