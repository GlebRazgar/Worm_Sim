import os
import yaml
import argparse
import imageio
import numpy as np

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
import torch
from model.Mnet import MNet

os.environ['CUDA_VISIBLE_DEVICES'] = "0" # Note: This might have no effect if CUDA is not available
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Determine the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")


    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='SegNeuron', help='path to config file')
    args = parser.parse_args()
    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    pth = '/Users/louisarge/Git/emulated_minds/SegNeuron/Train_and_Inference/SegNeuronModel.ckpt'

    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU='sub').to(device) # Use determined device
    checkpoint = torch.load(pth, map_location=device) # Load checkpoint to the determined device
    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        name = k.replace('module.', '') if 'module' in k else k
        new_state_dict[name] = v
    print('load mnet!')
    model.load_state_dict(new_state_dict)
    # model = model.to(device) # Model is already moved to device during initialization

    model.eval()
    # Run infernece
    image = imageio.imread('/Users/louisarge/Git/emulated_minds/SegNeuron/Train_and_Inference/data/em_data_highres.tif')
    inputs = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    inputs = inputs.to(device) # Use determined device
    print("inputs.shape: ", inputs.shape)
    # target = target.to(device) # Use determined device
    with torch.no_grad():
        pred, bound = model(inputs)
    print("pred.shape: ", pred.shape)
    print("bound.shape: ", bound.shape)
    output_img = np.squeeze(pred.data.cpu().numpy())
    print("output_img.shape: ", output_img.shape)  # (3, 20, 171, 171)
    
    # Transpose from (C, Z, Y, X) to (Z, Y, X, C)
    output_img = np.transpose(output_img, (1, 2, 3, 0))
    
    # Save as multi-channel 3D tiff stack
    imageio.volwrite('/Users/louisarge/Git/emulated_minds/SegNeuron/Train_and_Inference/data/em_data_highres_pred.tif', output_img)