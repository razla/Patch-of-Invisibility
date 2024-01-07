
import random
from typing import Tuple

import torch
from torchvision import transforms as T
from PIL import Image
    
    
def sample_square(
    min_h: int = 1, 
    max_h: int = 128, 
    min_w: int = 1, 
    max_w: int = 128,
    uniform_color_channels_flag: bool = False) -> torch.Tensor:
    h, w = random.randint(min_h, max_h), random.randint(min_w, max_w)
    square = torch.ones((3, h, w), dtype=torch.float) * random.random() \
        if uniform_color_channels_flag else \
        torch.stack([
            torch.ones((h, w), dtype=torch.float) * random.random(),
            torch.ones((h, w), dtype=torch.float) * random.random(),
            torch.ones((h, w), dtype=torch.float) * random.random()
        ])
    return square
    

def sample_strip(
    max_h: int = 128, 
    uniform_color_channels_flag: bool = False) -> torch.Tensor:
    return sample_square(max_h, max_h, 1, 1, uniform_color_channels_flag)
    
 
def init_square_attack_strips_patch_l_inf(
    patch_size: Tuple[int] = (128, 128), 
    uniform_color_channels_flag: bool = False) -> torch.Tensor:
    patch = torch.zeros(tuple([3] + list(patch_size)), dtype=torch.float)
    for x in range(patch_size[1]):
        patch[..., x] = sample_strip(patch_size[0], uniform_color_channels_flag)[:, :, 0]
    return patch

   
def update_square_attack_patch_l_inf(
    patch: torch.Tensor,
    uniform_color_channels_flag: bool = False) -> torch.Tensor:
    _, h, w = patch.shape
    square = sample_square(max_h=h, max_w=w, uniform_color_channels_flag=uniform_color_channels_flag)
    _, sh, sw = square.shape
    top = random.randint(0, h-sh)
    left = random.randint(0, w-sw)
    patch[:, top:top+sh, left:left+sw] = square
    return patch


def create_random_patch(num_iters: int = 3, uniform_color_channels_flag: bool = False) -> torch.Tensor:
    patch = init_square_attack_strips_patch_l_inf(uniform_color_channels_flag=uniform_color_channels_flag)
    for _ in range(num_iters):
        patch = update_square_attack_patch_l_inf(patch, uniform_color_channels_flag=uniform_color_channels_flag)
    return patch
