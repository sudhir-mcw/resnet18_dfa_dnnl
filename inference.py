import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from PIL import Image

import os
import json
import sys

"""
    Custom loader function to append prefix while loading npy files
"""


def load_npy(npy_file):
    npy_file = os.path.join("weights", npy_file)
    return np.load(npy_file)


if __name__ == "__main__":
    # create directories for inputs and output
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    # config file containing mean, std, arch, interpolation, input_shape for the model and classes
    CONFIG = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "arch": "resnet18",
        "interpolation": "bilinear",
        "input_shape": [3, 224, 224],
        "classes": [
            "tench",
            "English springer",
            "cassette player",
            "chain saw",
            "church",
            "French horn",
            "garbage truck",
            "gas pump",
            "golf ball",
            "parachute",
        ],
    }
    if len(sys.argv) != 2:
        print("Usage: python inference.py <input_image_path>")
        sys.exit(1)

    # Load input image and preprocess it
    input_image_path = sys.argv[1]
    image = Image.open(input_image_path)
    transform = transforms.Compose(
        [
            transforms.Resize(
                CONFIG["input_shape"][1:], interpolation=InterpolationMode.BILINEAR
            ),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(CONFIG["mean"], CONFIG["std"]),
        ]
    )
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    # pad the inputs with
    np.save("inputs/py_input.npy", input_tensor.detach().numpy())
    print("input shape", input_tensor.shape)

    """
        Convolution1 Starts
    """
    # Load weights and biases from the npy files
    conv1_wt = load_npy("conv1_wt.npy")
    bias1_wt = load_npy("conv1_bs.npy")

    conv1_wt_tensor = torch.tensor(conv1_wt, dtype=torch.float32)
    bias1_wt_tensor = torch.tensor(bias1_wt, dtype=torch.float32)

    conv = nn.Conv2d(
        in_channels=conv1_wt_tensor.shape[1],
        out_channels=conv1_wt_tensor.shape[0],
        kernel_size=conv1_wt_tensor.shape[2],
        stride=2,
        padding=3,
    )
    # Assign the loaded weights and biases to the convolutional layer
    conv.weight.data = conv1_wt_tensor
    conv.bias.data = bias1_wt_tensor

    # Perform convolution
    output_tensor = conv(input_tensor)
    print("conv output shape", output_tensor.shape)
    # Dump the output as npy
    np.save("outputs/py_output_conv1.npy", output_tensor.detach().numpy())
    """
        Convolution1 Ends
    """
    """
        Relu1 Starts
    """

    relu = nn.ReLU()

    output_tensor = relu(output_tensor)

    print("relu output shape", output_tensor.shape)
    # Dump the output as npy
    np.save("outputs/py_output_relu1.npy", output_tensor.detach().numpy())
    """
        Relu1 Ends
    """
