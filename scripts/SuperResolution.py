import argparse, os, sys, glob
import torchvision
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from taming.models import vqgan
from torch.autograd import Variable
import numpy as np
import torch
# from dataset import LRHRDataset
from torch.utils.data import DataLoader

from notebook_helpers import get_model, get_custom_cond, get_cond_options, get_cond, run

import ipywidgets as widgets
# import torchnvjpeg
import math


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="number of cpu threads to use during batch generation")
    opt = parser.parse_args()

    mode = widgets.Select(options=['superresolution'],
    value='superresolution', description='Task:')
    model = get_model(mode.value)
    custom_steps = 100

    # dir, options = get_cond_options(mode.value)
    # cond_choice_path = os.path.join(dir, 'sample_0.jpg')
    os.makedirs('outputs/SR', exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(opt.indir, "*.png")))
    for img in tqdm(zip(imgs)):
        img_name = os.path.split(img[0])[1]
        cond_choice_path = img[0]
        logs = run(model["model"], cond_choice_path, mode.value, custom_steps)
        sample = logs["sample"]
        sample = sample.detach().cpu()
        sample = torch.clamp(sample, -1., 1.)
        sample = (sample + 1.) / 2. * 255
        sample = sample.numpy().astype(np.uint8)
        sample = np.transpose(sample, (0, 2, 3, 1))
        print(sample.shape)
        Image.fromarray(sample[0]).save('outputs/SR/'+img_name)