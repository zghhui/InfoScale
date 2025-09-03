from diffusers.models.attention import Attention
import torch.nn.functional as F
import numpy as np
from numpy import mean
import torch
import os
import re
import cv2
import math
from einops import rearrange
from aux import Tref
from aux_sdxl import Tref as Tref_XL
import torch.fft as fft
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

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

def get_pca_feature(noise_pred_text, noise_pred_uncond, enable_gaussian=True):
    
    direction  = noise_pred_text - noise_pred_uncond
    # 高斯模糊
    # if enable_gaussian:
    # direction = gaussian_filter(direction, kernel_size=3, sigma=1.0)
    
    normalize_direction = (direction - direction.min()) / (direction.max() - direction.min()) * 2 - 1
    class_direction_score = normalize_direction[0].permute(1, 2, 0).view(-1, 4).cpu().numpy()
                    
    pca = PCA(n_components=1)
    pca.fit(class_direction_score)
    reduced = pca.transform(class_direction_score)[None, ...]
    return pca, reduced, direction.min(), direction.max() - direction.min()

def pca_fusion(last_pca_fea, noise_pred_text, noise_pred_uncond, alpha=0.2):
    pca, pca_fea, fea_min, fea_mean = get_pca_feature(noise_pred_text, noise_pred_uncond, False)
    # print(cosine_similarity(pca_fea[0], last_pca_fea[0]))
    device, dtype = noise_pred_text.device, noise_pred_text.dtype
 
    pca_fea_fusion = (1-alpha) * last_pca_fea + alpha * pca_fea
    pca_fea_fusion = pca.inverse_transform(pca_fea_fusion)
    pca_fea_fusion = pca_fea_fusion.reshape(noise_pred_text.shape)
    pca_fea_fusion = torch.tensor(pca_fea_fusion).to(device).to(dtype)
    
    pca_fea_fusion = (pca_fea_fusion + 1) / 2 * fea_mean + fea_min
    
    return pca_fea_fusion
    
def compute_entropy(attn_map, prompt_len):
    attn_map = attn_map[..., 1:prompt_len+1]  # 去掉首尾标记
    # attn_map *= 100
    #(h,w,tokens)
    attn_map = attn_map.mean(dim=-1)
    attn_map_sum = attn_map.sum(dim=(-1, -2), keepdim=True)
    # print(attn_map_sum)
    attn_map_normalized = attn_map / (attn_map_sum + 1e-9)
    attn_entropy = -torch.sum(attn_map_normalized * torch.log(attn_map_normalized + 1e-9))
    # print(attn_entropy.item())
    return attn_entropy.item()

def edit_noise(latents):
    b, c, h, w = latents.shape
    h1, h2 = h // 8 * 2, h // 8 * 6
    w1, w2 = w // 8 * 2, w // 8 * 6
    mask = torch.zeros_like(latents, dtype=latents.dtype)
    mask[:, :, h1:h2, w1:w2] = 1
    # latents[mask==0] = 0
    latents[mask==1] *= 0.99
    return latents

def freq_mix_3d(x, noise):
    # FFT    
    x_freq = fft.fftn(x, dim=(-2, -1)) 
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    noise_freq = fft.fftn(noise, dim=(-2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-2, -1))    
    
    LPF = box_low_pass_filter(x.shape, d_s=0.25, device=x.device)
    HPF = 1 - LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq * LPF + noise_freq_high
    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-2, -1)).real
    x_mixed = x_mixed.to(x.dtype)
    
    return x_mixed

def freq_mix_normal(x, noise):
    # FFT
    x_mean, x_std = x.mean(), x.std()
    x = (x - x_mean) / x_std
    noise_mean, noise_std = noise.mean(), noise.std()
    noise = (noise - noise_mean) / noise_std
    
    x_freq = fft.fftn(x, dim=(-2, -1)) 
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    noise_freq = fft.fftn(noise, dim=(-2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-2, -1))    
    
    LPF = box_low_pass_filter(x.shape, d_s=0.4, device=x.device)
    HPF = 1 - LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq * LPF + noise_freq_high
    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-2, -1)).real
    x_mixed = x_mixed.to(x.dtype)
    
    x_mixed = x_mixed * x_std + x_mean
    return x_mixed

def box_low_pass_filter(shape, d_s=0.25, device='cuda'):
    """
    Compute the ideal low pass filter mask (approximated version).

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
    """
    H, W = shape[-2], shape[-1]
    mask = torch.zeros(shape, device=device)
    if d_s==0:
        return mask

    threshold_s = round(int(H // 2) * d_s)

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0

    return mask    
    
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, log_t_n=1) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    torch.cuda.empty_cache()
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = torch.softmax(attn_weight, dim=-1) * log_t_n

    return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight


import seaborn as sns
import matplotlib.pyplot as plt
def atten_map_draw(attn_map, savename):
    sns.heatmap(attn_map, annot=False, cmap='viridis')
    os.makedirs('./image+attn_source_no', exist_ok=True)
    plt.savefig(f'./image+attn_source_no/{savename}.png', dpi=400)
    # import matplotlib as mpl
    # # we cannot use remote server's GUI, so set this  
    # import matplotlib.pyplot as plt
    # from matplotlib import cm as CM
    # from PIL import Image
    # import numpy as np
    # import cv2

    # img_path = "/home/mlsnrs/data/zyz/zgh/AIGC/High_resulation/paper_experiments/attn_entropy_adaptive/logging-dir/36_seed123.jpg"
    # img = cv2.imread(img_path)
    # h, w, c = img.shape
    # print(h, w, c)
    
    # attn_map = attn_map.astype(np.float32)
    # print(attn_map.shape)
    # attn_map = cv2.resize(attn_map, (w, h))
    # # adaptive gaussian filter
    # # heatmap = attn_map/np.max(attn_map)
    # # must convert to type unit8
    # heatmap = np.uint8(255 * attn_map)
    # print(heatmap)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # superimposed_img2 = heatmap

    # cv2.imwrite(f'image+attn/{savename}.jpg', superimposed_img2)  
class Attn_CFG:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, prompt):
        # self.save_path_dir = save_path_dir
        self.entropy_text = [[] for _ in range(len(prompt))]
        self.entropy_uncond = [[] for _ in range(len(prompt))]
        self.entropy_uncond_block = [[] for _ in range(len(prompt))]
        self.entropy_text_block = [[] for _ in range(len(prompt))]
        self.attention_map_mask = []
        self.step = -1
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
        if n != 77:
            T = list(Tref[attn.name+'.processor'])
            assert len(T) == 1
            T = T[0]
            # 每个注意力头的维数
            dim_head = attn.inner_dim / attn.heads

            qk_scale = ((math.log(sequence_length, T)) / dim_head) ** 0.5   
        if n == 77:
            # b, h, n, d = query.shape
            b, h, n, _ = value.shape
            value_ = torch.eye(n, device=value.device, dtype=value.dtype).unsqueeze(0).unsqueeze(0).expand(b, h, n, n)
            
            attention_probs,_= scaled_dot_product_attention(
                query, key, value_, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                scale=qk_scale
            )
            attention_probs = rearrange(
                attention_probs,
                'batch attn_head (h w) attn_dim -> batch attn_head h w attn_dim ',
                h=int(np.sqrt(attention_probs.shape[-2]))
            ) 
            # torch.Size([b*2, 5, 64, 64, 77])
            attention_probs = attention_probs.mean(dim=1)
            # torch.Size([b*2, 64, 64, 77])
            att_uncond, att_text = attention_probs.chunk(2)
            b = att_uncond.shape[0]
            
            entropy_uncond = [[] for _ in range(b)]
            entropy_text = [[] for _ in range(b)]
            if attn.name == 'down_blocks.0.attentions.0.transformer_blocks.0.attn2':
                self.step += 1
            for i in range(b):
                entropy_uncond[i].append(compute_entropy(att_uncond[i], self.prompt_len[i]))
                entropy_text[i].append(compute_entropy(att_text[i], self.prompt_len[i]))
                # if i == 3 and attn.name=='up_blocks.2.attentions.2.transformer_blocks.0.attn2':
                #     import seaborn as sns
                #     import matplotlib.pyplot as plt
                #     # 绘制热力图
                
                #     attn_map = att_text[i,..., 1:4].mean(dim=-1)
                #     plt.figure(figsize=(10, 8))
                #     attn_map = attn_map.cpu().numpy()
                #     atten_map_draw(attn_map, f'{i}_{self.step}_{attn.name}')
                # sns.heatmap(attn_map.cpu().numpy(), cmap="viridis", annot=False)

                # # 添加标题和标签
                # plt.title("Attention Map Heatmap")
                # plt.xlabel("Query Index")
                # plt.ylabel("Key Index")     
                # os.makedirs('./attention_map_sd', exist_ok=True)
                #             

            if attn.name == 'up_blocks.3.attentions.2.transformer_blocks.0.attn2':
                for i in range(b):
                    self.entropy_uncond[i].append(mean(entropy_uncond[i]))
                    self.entropy_text[i].append(mean(entropy_text[i]))
            elif attn.name == 'up_blocks.3.attentions.0.transformer_blocks.0.attn2':
                h, w = attention_probs.shape[-3], attention_probs.shape[-2]
                att_text = att_text[..., 1:]
                self.attention_map_mask = []
                for i in range(b):
                    att_text_item = att_text[i,..., :self.prompt_len[i]].mean(dim=-1)
                    self.attention_map_mask.append(att_text_item > att_text_item.mean(dim=[-1, -2]))
                    # 倒数第二个block作为语义的衡量
                    self.entropy_uncond_block[i].append(entropy_uncond[i][-1])
                    self.entropy_text_block[i].append(entropy_text[i][-1])                                  
        if qk_scale != None:
            log_t_n=0.75
        else:
            log_t_n=1
        # if self.step > 45:
        #     log_t_n=1
        hidden_states,_ = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            scale=qk_scale, log_t_n=log_t_n
        )
        print(123)
        print(attn.rescale_output_factor)

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

        return hidden_states


class Attn_CFG_XL:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, prompt):
        # self.save_path_dir = save_path_dir
        self.entropy_text = [[] for _ in range(len(prompt))]
        self.entropy_uncond = [[] for _ in range(len(prompt))]
        self.entropy_uncond_block = [[] for _ in range(len(prompt))]
        self.entropy_text_block = [[] for _ in range(len(prompt))]
        self.attention_map_mask = []
        self.step = -1
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
        if attn.name == 'down_blocks.1.attentions.0.transformer_blocks.0.attn1':
            self.step += 1
        if n != 77:
            T = list(Tref_XL[attn.name+'.processor'])
            assert len(T) == 1
            T = T[0]
            # 每个注意力头的维数
            dim_head = attn.inner_dim / attn.heads

            qk_scale = ((math.log(sequence_length, T)) / dim_head) ** 0.5   
        if n == 77 and False:
            # b, h, n, d = query.shape
            b, h, n, _ = value.shape
            value_ = torch.eye(n, device=value.device, dtype=value.dtype).unsqueeze(0).unsqueeze(0).expand(b, h, n, n)
            
            attention_probs,_ = scaled_dot_product_attention(
                query, key, value_, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                scale=qk_scale
            )
            attention_probs = rearrange(
                attention_probs,
                'batch attn_head (h w) attn_dim -> batch attn_head h w attn_dim ',
                h=int(np.sqrt(attention_probs.shape[-2]))
            ) 
            # torch.Size([b*2, 5, 64, 64, 77])
            attention_probs = attention_probs.mean(dim=1)
            # torch.Size([b*2, 64, 64, 77])
            att_uncond, att_text = attention_probs.chunk(2)
            b = att_uncond.shape[0]
            
            entropy_uncond = [[] for _ in range(b)]
            entropy_text = [[] for _ in range(b)]
            for i in range(b):
                entropy_uncond[i].append(compute_entropy(att_uncond[i], self.prompt_len[i]))
                entropy_text[i].append(compute_entropy(att_text[i], self.prompt_len[i]))
                
                # # ##绘制热力图
                # import seaborn as sns
                # import matplotlib.pyplot as plt
                # # 绘制热力图
                # attn_map = att_text[i,..., :self.prompt_len[i]].mean(dim=-1)
                # plt.figure(figsize=(10, 8))
                # sns.heatmap(attn_map.cpu().numpy(), cmap="viridis", annot=False)

                # # 添加标题和标签
                # plt.title("Attention Map Heatmap")
                # plt.xlabel("Query Index")
                # plt.ylabel("Key Index")     
                # plt.savefig(f'./attention_map_sdxl/{self.step}_{attn.name}.png')
            if attn.name == 'up_blocks.0.attentions.0.transformer_blocks.8.attn2':
                for i in range(b):
                    self.entropy_uncond[i].append(entropy_uncond[i][-1])
                    self.entropy_text[i].append(entropy_text[i][-1])
            elif attn.name == 'up_blocks.1.attentions.2.transformer_blocks.1.attn2':
                h, w = attention_probs.shape[-3], attention_probs.shape[-2]
                att_text = att_text[..., 1:]
                self.attention_map_mask = []
                for i in range(b):
                    att_text_item = att_text[i,..., :self.prompt_len[i]].mean(dim=-1)
                    
                    self.attention_map_mask.append(att_text_item > att_text_item.mean(dim=[-1, -2]))
                    # 倒数第二个block作为语义的衡量
                    self.entropy_uncond_block[i].append(entropy_uncond[i][-1])
                    self.entropy_text_block[i].append(entropy_text[i][-1])                                  
        if qk_scale != None:
            log_t_n=0.75
        else:
            log_t_n=1
        hidden_states,_ = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            scale=qk_scale,log_t_n=log_t_n
        )
        # print(attn.rescale_output_factor)
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

        return hidden_states
    
def attention_name(pipe):
    for name, module in pipe.unet.named_modules():
        if isinstance(module, Attention):
            module.name = name
    return pipe