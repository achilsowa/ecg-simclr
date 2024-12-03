"""Modified from IJEPA code, and Mourad adaptation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

import numpy as np
from functools import partial
import math


from src.utils.tensors import (
    trunc_normal_,
)


LARGE_NUMBER = 1e5

def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        #pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        pos_embed = np.concatenate([np.random.randn([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
        

class Attention(nn.Module):
    """
    Multi head attention block
    """
    def __init__(self, head_size, num_heads, att_drop=0., proj_drop=0., qkv_bias=False, qk_scale=None,  mask=False) -> None:
        super().__init__()
        dmodel = head_size * num_heads
        self.head_size = head_size
        self.num_heads = num_heads
        self.mask = torch.tril(torch.ones(dmodel, dmodel)) if mask else 1
        self.scale = qk_scale or head_size ** -0.5
        self.attn_drop = nn.Dropout(att_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.qkv = nn.Linear(dmodel, 3*dmodel, bias=qkv_bias)
        self.proj = nn.Linear(dmodel, dmodel)


    def forward(self, x):
        """Evaluate multi head attention
        param x (torch.FloatTensor of shape (b, n, nh*hs))
            input tensor containing either model input or output of the previous block.
        return (torch.FloatTensor of shape (b, n, nh*hs))
        """
        qkv = rearrange(qkv, 'b n (grp nh hs) -> grp b nh n hs', grp=3, nh=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        mask_att = self.mask * self.scale * (q @ k.transpose(-2, -1)) + (self.mask -1 ) * LARGE_NUMBER
        att = self.attn_drop(mask_att.softmax(dim=-1))
        
        x = rearrange(att @ v, 'b nh n hs -> b n (nh hs)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, att


class MLP(nn.Module):
    """Point Wise MLP"""
    def __init__(self, dmodel, hidden_size, act_fct=nn.GELU, drop=0. ) -> None:
        super().__init__()
        self.act_fct = act_fct()
        self.dmodel = dmodel
        self.hidden_size = hidden_size
        self.fc1 = self.w1 = nn.Linear(dmodel, hidden_size)
        self.fc2 = self.w2 = nn.Linear(hidden_size, dmodel)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        """Pointwise mlp
        @param x (torch.FloatTensor of shape (batch_size, seq_length, dmodel)) 
            input tensor containing output of previous layernorm(x+attention(x))
        @return output applied at each (batch_size, seq_length) elements
        """
        y = self.act_fct(self.w1(x))
        y = self.dropout(y)
        y = self.w2(y)
        y = self.dropout(y)
        return y
        


class Block(nn.Module):
    """Transformer block
    """
    def __init__(self, dim, num_heads, 
                attn_drop=0., 
                proj_drop=0., 
                mask=False, 
                qkv_bias=False, 
                qk_scale=None,
                mlp_ratio=4.,
                mlp_drop=0.,
                drop_path=0.,
                act_fct=nn.GELU,
                norm_layer=nn.LayerNorm
              ) -> None:
        super().__init__()
        self.attn = Attention(dim//num_heads, num_heads, attn_drop, proj_drop, qkv_bias, qk_scale, mask)
        self.mlp = MLP(dim, int(dim*mlp_ratio), act_fct, mlp_drop)
        self.drop_path =  nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """One transformer block. Norm -> Sublayer -> Residual connection, as in vit paper
        @param x (torch.FloatTensor of shape (batch_size, seq_length, dmodel))
        return output after following all the operations in transformer
        """
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward_(self, x):
        """One transformer block. Sublayer -> Residual connection -> Norm, as in original transformer paper
        @param x (torch.FloatTensor of shape (batch_size, seq_length, dmodel))
        return output after following all the operations in transformer
        """
        y, attn = self.attn(x)
        x = self.norm1(x + y)
        x = self.norm2(x + self.mlp(x))
        return x
    

class Transformer (nn.Module):
    """ Transformer """
    def __init__(
        self,
        embed_dim,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        mask=False,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        add_positional_embedding=True,
        **kwargs
    ):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                mlp_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, mask=mask)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.add_positional_embedding = add_positional_embedding

        
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()
        
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        # -- fwd prop
        def add_embedding():
            if self.add_positional_embedding:
                b, n, d = x.shape
                pos_embed = torch.from_numpy(get_1d_sincos_pos_embed(d, n)).float()
                pos_embed = rearrange(pos_embed, 'n d -> 1 n d')
        
                x = x + pos_embed
                return x
            else: return x
        
        x = add_embedding()
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        return x
  

class PatchGroupEmbed(nn.Module):
    """ 1D signal to Patch Embedding. 
    Consider all in_chans/leads together
    """
    def __init__(self, signal_size=2500, patch_size=16, in_chans=12, embed_dim=768):
        super().__init__()
        assert signal_size % patch_size == 0, "num_patches should be signal_size // patch_size"
        num_patches = (signal_size // patch_size)
        self.signal_size = signal_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """inputs 
        @params x(torch.FloatTensor of shape (batch_size, n_leads, signal_size))
        """
        y = self.proj(x).transpose(1, 2) # -> (batch_size, num_patches, embed_dim)
        return y

class PatchIndependentEmbed(nn.Module):
    """ 1D signal to Patch Embedding
    Consider all in_chans/leads independently
    """
    def __init__(self, signal_size=2500, patch_size=16, in_chans=12, embed_dim=768):
        super().__init__()
        assert signal_size % patch_size == 0, "num_patches should be in_chans * signal_size // patch_size"
        num_patches = in_chans * (signal_size // patch_size)
        self.signal_size = signal_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.proj = nn.Conv1d(1, in_chans * embed_dim, kernel_size=patch_size, stride=patch_size)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
            
    def forward(self, x):
        """inputs 
        @params x(torch.FloatTensor of shape (batch_size, n_leads, signal_size))
        """
        B, C, W = x.shape
        x = x.reshape(B, 1, C*W)
        y = self.proj(x).reshape(B, self.embed_dim, self.num_patches).transpose(1, 2) # -> (batch_size, num_patches, embed_dim)
        return y


class DownsamplingEmbed(nn.Module):
    """ Ecg downsampling
    Here we downsample ecg with non learnable kernels. Each leads/in_chans is handled independently
    and so we take (B, C, W) and return (B, C, E), only downsampling from W to E(embed_dim)
    The downsampling can either use mean, or take one value at a time
    """
    def __init__(self, signal_size=2500, patch_size=16, in_chans=12, embed_dim=768, use_mean=False, learn_weight=False):
        super().__init__()
        assert signal_size % patch_size == 0, "patchembedding size must be signal_size // patch_size"
        patch_size = signal_size // embed_dim
        self.signal_size = signal_size
        self.patch_size = patch_size
        self.num_patches = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv1d(1, in_chans , kernel_size=patch_size, stride=patch_size)
        nn.init.ones_(self.proj.weight) if use_mean else nn.init.dirac_(self.proj.weight)
        self.proj.requires_grad_ = learn_weight

        
    def forward(self, x):
        """inputs 
        @params x(torch.FloatTensor of shape (batch_size, n_leads, signal_size))
        """
        B, C, W = x.shape
        
        x = x.reshape(B, 1, C*W)
        y = self.proj(x).reshape(B, -1, self.num_patches).transpose(1, 2) # -> (batch_size, num_patches, embed_dim)
        return y


class PatchFlattenedStaticEmbed(nn.Module):
    """Consider (12, 2500) leads ECG as 1D signal of length 12x2500. 
    Split the signal in num_patches blocks each of dim == patch_size.
    Nothing is learned in this case
    """
    def __init__(self, signal_size=2500, patch_size=625, in_chans=12, embed_dim=625, learnable=False):
        super().__init__()
        assert patch_size == embed_dim, "expected patch_size == embed_dim"
        self.num_patches = in_chans * (signal_size // patch_size)
        self.signal_size = signal_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim    
        
            
    def forward(self, x):
        """inputs 
        @params x(torch.FloatTensor of shape (b, n, w))
        @return (torch.FloatTensor of shape (b, np, d))
        """
        x = rearrange(x, 'b c (np ps)-> b (c np) ps', ps=self.patch_size)
        return x

class PatchFlattenedEmbed(nn.Module):
    """Consider (12, 2500) leads ECG as 1D signal with 12 channels
    To construct block:
        1. Take a portion of size ps of the original of the original signal -> (c, ps)
        2. Flatten that portion to obtain a 1d signal (ps * c)
        3. Project that using a learnable map on smaller embed_dim dimensional space
    """
    def __init__(self, signal_size=2500, patch_size=25, in_chans=12, embed_dim=625):
        super().__init__()
        assert patch_size % signal_size == 0, "expected patch_size to divide signal_size"
        self.num_patches =  signal_size // patch_size
        self.signal_size = signal_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim    
        patch_dim = in_chans * patch_size
        self.proj = nn.Sequential(
            Rearrange('b c (np ps) -> b np (ps c)', ps=self.patch_size),
            nn.LayerNorm(patch_dim), 
            nn.Linear(patch_size, embed_dim),
            nn.LayerNorm(embed_dim)
            )

    def forward(self, x):

        """inputs 
        @params x(torch.FloatTensor of shape (b, n, w))
        @return (torch.FloatTensor of shape (b, np, d))
        """
        x = self.proj(x)
        return x

   



class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        signal_size=2500,
        patch_size=10,
        in_chans=12,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        cls_token=False,
        mask=False,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        # --
        self.patch_embed = PatchFlattenedEmbed(
            signal_size=signal_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches + (1 if cls_token else 0)
        # --
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=cls_token)
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            self.patch_embed.num_patches,
                                            cls_token=cls_token)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        if cls_token: self.cls_token = self.pos_embed[:, 0]
        # --
        self.transformer = Transformer(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, 
                                       attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, mask=mask, norm_layer=norm_layer,
                                       init_std=init_std, add_positional_embedding=False)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # -- patchify x
        x = self.patch_embed(x)
        b, n, d = x.shape

        # -- add positional embedding to x (self.pos_embed[:, 0] correspond to [cls])        
        if self.cls_token:
            x = torch.concat([torch.zeros(b, 1, d, device=x.device), x], dim=1)
        x = x + self.pos_embed

        # -- fwd prop
        x = self.transformer(x)
        
        return x


def ecgt_tiny(patch_size=25, **kwargs):
    """Constructs a Vision Transformer for ECG signals.
    @param: patch_size (int): The size of each patch.
    @param: **kwargs: Additional arguments to be passed to the VisionTransformer constructor.
    @return:
        VisionTransformer: A Vision Transformer model for processing ECG signals.
    """
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), cls_token=True, **kwargs)
    return model


def ecgt_small(patch_size=25, **kwargs):
    """Constructs a Vision Transformer for ECG signals.
    @param: patch_size (int): The size of each patch.
    @param: **kwargs: Additional arguments to be passed to the VisionTransformer constructor.
    @return:
        VisionTransformer: A Vision Transformer model for processing ECG signals.
    """
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def ecgt_base(patch_size=25, **kwargs):
    """Constructs a Vision Transformer for ECG signals.
    @param: patch_size (int): The size of each patch.
    @param: **kwargs: Additional arguments to be passed to the VisionTransformer constructor.
    @return:
        VisionTransformer: A Vision Transformer model for processing ECG signals.
    """
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model