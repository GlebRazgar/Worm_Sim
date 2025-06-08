import os
import yaml
import argparse
import imageio
import numpy as np
from src.affinity_postprocessing import segment_neuron_affinities, segment_to_image

# Patch for collections.Mapping import error in Python >= 3.10
import sys
import collections

if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    import collections.abc

    collections.Mapping = collections.abc.Mapping
    collections.MutableMapping = collections.abc.MutableMapping
    collections.Sequence = collections.abc.Sequence
# End of patch

from attrdict import AttrDict
from collections import OrderedDict
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")

import torch
from src.mnet import MNet

if __name__ == "__main__":
    # Determine the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    pth = "models/SegNeuronModel.ckpt"
    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub").to(
        device
    )  # Use determined device
    checkpoint = torch.load(
        pth, map_location=device
    )  # Load checkpoint to the determined device
    new_state_dict = OrderedDict()
    state_dict = checkpoint["model_weights"]
    for k, v in state_dict.items():
        name = k.replace("module.", "") if "module" in k else k
        new_state_dict[name] = v
    print("load mnet!")
    model.load_state_dict(new_state_dict)
    # model = model.to(device) # Model is already moved to device during initialization

    model.eval()
    # Run infernece
    image = imageio.imread(
        "/Users/louisarge/Git/emulated_minds/SegNeuron/Train_and_Inference/data/em_data_highres.tif"
    )
    inputs = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    inputs = inputs.to(device)  # Use determined device
    print("inputs.shape: ", inputs.shape)
    # target = target.to(device) # Use determined device
    with torch.no_grad():
        affinity, mask = model(inputs)
    print("affinity.shape: ", affinity.shape)
    print("mask.shape: ", mask.shape)
    affinity = np.squeeze(affinity.data.cpu().numpy())
    mask = np.squeeze(mask.data.cpu().numpy())
    print("affinity.shape: ", affinity.shape)  # (3, 20, 171, 171)
    imageio.volwrite(
        "output/em_data_highres_aff.tif",
        np.transpose(affinity, (1, 2, 3, 0)),
    )
    print("mask.shape: ", mask.shape)  # (1, 20, 171, 171)
    imageio.volwrite("output/em_data_highres_mask.tif", mask.unsqueeze(0))

    # Transpose from (C, Z, Y, X) to (Z, Y, X, C)

    segment = segment_neuron_affinities(affinity, mask)
    print("segment.shape: ", segment.shape)
    segment_img = segment_to_image(segment)
    print("segment_img.shape: ", segment_img.shape)

    # Save as multi-channel 3D tiff stack
    imageio.volwrite(
        "output/em_data_highres_seg.tif",
        segment_img,
    )
