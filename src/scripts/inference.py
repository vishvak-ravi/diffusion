import torch
import numpy as np
import torchvision.utils as vutils

from models import UNet
from sampling import DDPM_sampling
from configs import Config

CONFIG_PATH = "src/configs/cifar10_ddpm_single_class.json"
WEIGHTS_PATH = "weights/cifar10_ddpm_single_class.pth"
OUTPUT_DIR = "output_images"

if __name__ == "__main__":
    config = Config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    model = UNet(config)
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    samples, reverse_path = DDPM_sampling(model, config, track_path=128)

    sample_grid = vutils.make_grid(samples, nrow=10, padding=2)
    sample_path = vutils.make_grid(reverse_path, nrow=10, padding=2)
    vutils.save_image(sample_grid, "samples/samples.jpg")
    vutils.save_image(sample_path, "samples/reverse_trajectory.jpg")
