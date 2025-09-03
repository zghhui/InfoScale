# adopted from
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
# and
# https://github.com/TianxingWu/FreeInit/blob/master/freeinit_utils.py
# and
# https://github.com/AILab-CVC/FreeNoise
# thanks!


import os
import math
import torch

import numpy as np
import torch.nn as nn

from einops import rearrange, repeat
import  torch.fft as fft
from diffusers.models.attention import Attention
import torch.nn.functional as F
from aux_sdxl import Tref

from einops import rearrange


#################################################################################
#                                  Long Video Utils                                   #
#################################################################################

def gaussian_kernel(kernel_size=3, sigma=1.0, channels=3):
    x_coord = torch.arange(kernel_size)
    gaussian_1d = torch.exp(-(x_coord - (kernel_size - 1) / 2) ** 2 / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
    kernel = gaussian_2d[None, None, :, :].repeat(channels, 1, 1, 1)
    
    return kernel

def gaussian_filter(latents, kernel_size=3, sigma=1.0):
    channels = latents.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, channels).to(latents.device, latents.dtype)
    blurred_latents = F.conv2d(latents, kernel, padding=kernel_size//2, groups=channels)
    
    return blurred_latents

def gaussian_2d(x, y, mx, my, sx, sy, dtype):
    return torch.exp(
        -(((x - mx) ** 2) / (2 * sx ** 2) + ((y - my) ** 2) / (2 * sy ** 2))
    ).to(dtype)

# 创建高斯加权函数
def gaussian_weight_noise(latents, KERNEL_DIVISION=3.0):
    height, width = latents.shape[-2], latents.shape[-1]
    device = latents.device
    dtype = latents.dtype
    threshold = 1.05
    x = torch.linspace(0, height - 1, height)
    y = torch.linspace(0, width - 1, width)
    x, y = torch.meshgrid(x, y, indexing="ij")
    noise_patch = gaussian_2d(x, y, mx=int(height / 2), my=int(width / 2),
                              sx=float(height / KERNEL_DIVISION), sy=float(width / KERNEL_DIVISION), dtype=dtype)
    min_val = torch.min(noise_patch)
    max_val = torch.max(noise_patch)
    noise_patch_normalized = 1 + (noise_patch - min_val) * (threshold - 1) / (max_val - min_val)
    noise_patch_normalized = 2 - noise_patch_normalized
    noise_patch_normalized = noise_patch_normalized.to(device)
    edit_latents = latents * noise_patch_normalized
    return edit_latents

def gaussian_low_pass_filter(shape, d_s=0.25):
    """
    Compute the Gaussian low pass filter mask using vectorized operations, ensuring exact
    calculation to match the old loop-based implementation.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
    """
    H, W = shape[-2], shape[-1]
    if d_s == 0:
        return torch.zeros(shape)

    # Create normalized coordinate grids for T, H, W
    # Generate indices as in the old loop-based method
    h = torch.arange(H).float() * 2 / H - 1
    w = torch.arange(W).float() * 2 / W - 1
    
    # Use meshgrid to create 3D grid of coordinates
    grid_h, grid_w = torch.meshgrid(h, w, indexing='ij')

    # Compute squared distance from the center, adjusted for the frequency cut-offs
    d_square = ((grid_h * (1 / d_s)).pow(2) + (grid_w * (1 / d_s)).pow(2))

    # Compute the Gaussian mask
    mask = torch.exp(-0.5 * d_square)

    return mask

def gaussian_high_pass_filter(shape, d_s=0.75):
    """
    Compute the Gaussian low pass filter mask using vectorized operations, ensuring exact
    calculation to match the old loop-based implementation.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
    """
    H, W = shape[-2], shape[-1]
    if d_s == 0:
        return torch.zeros(shape)

    # Create normalized coordinate grids for T, H, W
    # Generate indices as in the old loop-based method
    h = torch.arange(H).float() * 2 / H - 1
    w = torch.arange(W).float() * 2 / W - 1
    
    # Use meshgrid to create 3D grid of coordinates
    grid_h, grid_w = torch.meshgrid(h, w, indexing='ij')

    # Compute squared distance from the center, adjusted for the frequency cut-offs
    d_square = ((grid_h * (1 / d_s)).pow(2) + (grid_w * (1 / d_s)).pow(2))

    # Compute the Gaussian mask
    mask = torch.exp(-0.5 * d_square)
    mask = 1-mask

    return mask


def freq_mix_3d(x, noise, d_s=0.25, high_d_s=0.5, t=0):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    noise_freq = fft.fftn(noise, dim=(-2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-2, -1))
    
    # frequency mix
    C, H, W = x.shape
    filters = gaussian_low_pass_filter((H, W), d_s=d_s).to(x.device)
    # high_d_s = 1 - 0.5*mix_alpha
    high_filters = gaussian_high_pass_filter((H, W), d_s=high_d_s).to(x.device)
    low_cut  = filters.unsqueeze(0).repeat(C, 1, 1)
    high_cut  = high_filters.unsqueeze(0).repeat(C, 1, 1)
    
    LPF = low_cut
    # LPF = torch.max(low_cut, high_cut)
    HPF = 1 - LPF
    if t < 15:
        x_freq_low = x_freq * LPF
        noise_freq_high = noise_freq * HPF
    else:
        x_freq_low = x_freq * HPF
        noise_freq_high = x_freq * LPF
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-2, -1)).real
    
    # x_mixed = x_mixed * mix_alpha + x * (1 - mix_alpha) # mix in spatial domain

    return x_mixed

def attention_name(pipe):
    for name, module in pipe.unet.named_modules():
        if isinstance(module, Attention):
            module.name = name
    return pipe

def downsample_upsample(x, scale=2):
    # x: (B, H, W, D)
    B, H, W, D = x.shape
    # 下采样（假设双线性插值），先转成 BCHW
    x_ = x.permute(0, 3, 1, 2)  # (B, D, H, W)
    x_down = F.interpolate(x_, scale_factor=1/scale, mode='bilinear', align_corners=False)
    # 上采样回原分辨率
    x_up = F.interpolate(x_down, size=(H, W), mode='bilinear', align_corners=False)
    x_up = x_up.permute(0, 2, 3, 1)  # (B, H, W, D)
    return x_up

class Attn_CFG:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, prompt):
        # self.save_path_dir = save_path_dir
        self.entropy_text = [[] for _ in range(len(prompt))]
        self.entropy_uncond = [[] for _ in range(len(prompt))]
        self.enable_dilate = 1
        self.prompt_len = [min(78, len(prompt[i].split(' '))+1) for i in range(len(prompt))]
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads     
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1

        b, h, n, _ = value.shape
        qk_scale = None
        # 不使用Attn
        query_clone = query.clone()
        key_clone = key.clone()
        value_clone = value.clone()
        
        T = list(Tref[attn.name+'.processor'])
        assert len(T) == 1
        T = T[0]
        outlayer = True if T == 16384 else False
        selfattn = True if T != 77 else False
        ## 交叉注意力不用，最外层直接使用，剩下层高低频融合
        # if not outlayer:
        #     # 每个注意力头的维数
        #     dim_head = attn.inner_dim / attn.heads

        #     qk_scale = ((math.log(sequence_length, T)) / dim_head) ** 0.5
            
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            scale=qk_scale
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        # 执行Attn
        if selfattn and not outlayer:
            T = list(Tref[attn.name+'.processor'])
            assert len(T) == 1
            T = T[0]
            # 每个注意力头的维数
            dim_head = attn.inner_dim / attn.heads

            qk_scale = ((math.log(sequence_length, T)) / dim_head) ** 0.5   
        
            ## 使用Attn
            hidden_states_attn = F.scaled_dot_product_attention(
                query_clone, key_clone, value_clone, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                scale=qk_scale
            )
            hidden_states_attn = hidden_states_attn.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states_attn = hidden_states_attn.to(query.dtype)
            
            # linear proj
            hidden_states_attn = attn.to_out[0](hidden_states_attn)

            # dropout
            hidden_states_attn = attn.to_out[1](hidden_states_attn)

            if input_ndim == 4:
                hidden_states_attn = hidden_states_attn.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states_attn = hidden_states_attn + residual

            hidden_states_attn = hidden_states_attn / attn.rescale_output_factor
            # 高低频融合
            latent_h = int(hidden_states.shape[1] ** 0.5)
            hidden_states = rearrange(hidden_states, 'bh (h w) d -> bh h w d', h=latent_h)
            hidden_states_attn = rearrange(hidden_states_attn, 'bh (h w) d -> bh h w d', h=latent_h)

            # 用下采样-上采样获得低频分量
            low_freq = downsample_upsample(hidden_states)
            low_freq_attn = downsample_upsample(hidden_states_attn)

            # 高频分量 = 原始 - 低频
            high_freq = hidden_states - low_freq
            low_freq_attn = low_freq_attn

            # 按照你原来的融合方式
            hidden_states = high_freq + low_freq_attn

            hidden_states = rearrange(hidden_states, 'bh h w d -> bh (h w) d')
        return hidden_states