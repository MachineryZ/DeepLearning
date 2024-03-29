# https://blog.csdn.net/qq_44941689/article/details/127064266
# https://blog.csdn.net/Peach_____/article/details/128663957
# https://blog.csdn.net/weixin_43850253/article/details/128274577
# https://yuezhou-oh.github.io/blog/paperreading/Understanding_diffusion_model.html
import math
from inspect import isfunction
from functools import partial

# %matplotlib inline

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    else:
        return d() if isfunction(d) else d



# ================================================================================
# model part the end

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s ) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schdule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

# 创建图实例
import numpy as np
x = np.linspace(1, 1001, 1000)
timesteps = 1000
fig, ax = plt.subplot()
ax.plot(x, (cosine_beta_schedule(timesteps, s=0.008)/50).numpy(), label="cosine")
ax.plot(x, linear_beta_schedule(timesteps).numpy(), label="linear")
ax.plot(x, quadratic_beta_schdule(timesteps).numpy(), label="quadratic")
ax.plot(x, sigmoid_beta_schedule(timesteps).numpy(), label="sigmoid")
plt.legend()
plt.show()


# betas = betas
# alphas = 1 - betas
# alphas_cumprod alpha_t = alphas_t = alphas_1 * alphas_2 * ... *  alpha_t
# alphas_cumpord_prev = alphas_cumprod alpha_t-1
# sqrt_alphas_cumprod = sqrt(alpha_t)
# sqrt_one_minus_alphas_cumprod = sqrt(1 - alpha_t)
# posterior_variance beta * (1 - alpha_t-1) / (1 - alpha_t)


timesteps = 200

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

posterior_variance = betas * (1 - alphas_cumprod_prev) / (1. - alphas_cumprod)

# 累乘的一个函数
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# ================================================================================

# xt = sqrt(alpha_t) x0 + sqrt(1 - alpha_t) epsilon

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod_t, t, x_start.shape)
    sqrt_one_minus_alpha_cumprod_t = extract(sqrt_one_minus_alpha_cumprod_t, t, x_start.shape)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

image_size = 128
transform = Compose([
    Resize(image_size), # 变为形状为128*128
    CenterCrop(image_size), # 中心裁剪
    ToTensor(), # turn into Numpy array of shape HWC, divide by 255
    Lambda(lambda t: (t * 2) - 1), # 变为[-1,1]范围
    
])

x_start = transform(image).unsqueeze(0)
x_start.shape

import numpy as np
reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])

def get_noisy_image(x_start, t):
    x_noisy = q_sample(x_start, t=t)
    noisy_image = reverse_transform()
    return noisy_image

t = torch.tensor([40])
get_noisy_image(x_start, t)

# pytorch 官方的一个画图函数
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
# image

import matplotlib.pyplot as plt
torch.manual_seed(0)
def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        imgs = [imgs]
    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200, 200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        if with_orig:
            axs[0, 0].set(title="Original image")
            axs[0, 0].title.set_size(8)
        if row_title is not None:
            for row_idx in range(num_rows):
                axs[row_idx, 0].set(ylabel=row_title[row_idx])
        
        plt.tight_layout()
        plt.show()

def p_losses(denoise_model, x_start, t, nois=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alpha_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

def p_sample_loop(model, shape):
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device = device)
    imgs = []
    for i in tqdm(reversed(range(0, timesteps)), desc="sampling loop time step", total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

from pathlib import Path
def num_to_groups(num)


