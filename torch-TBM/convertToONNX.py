import torch
import torch.nn as nn

def onnx_factory(model: nn.Module, output_name: str = "./vgg11.onnx"):
    torch.onnx.export(model, torch.rand(1, 3, 224, 224), output_name, export_params=True, verbose=True)

if __name__ == "__main__":
    from vgg_11 import VGG11

    path = "./vgg11_epoch_20.pth.tar"
    model = VGG11(3, 3)
    param = torch.load(path)
    model.load_state_dict(param["state_dict"])
    onnx_factory(model)
    