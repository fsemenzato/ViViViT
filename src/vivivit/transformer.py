import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import torch.utils.checkpoint as checkpoint

class CircularConv3d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=0)

    def forward(self, x):
        x = F.pad(x, [self.pad] * 6, mode='circular')
        return self.conv(x)

class PixelShuffle3D(nn.Module):

    def __init__(self, upscale_factor):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x):
        r = self.r
        (B, C_in, T, H, W) = x.shape
        C = C_in // r ** 3
        x = x.view(B, C, r, r, r, T, H, W)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(B, C, T * r, H * r, W * r)
        return x

class ResBlock3d(nn.Module):

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = CircularConv3d(channels, channels, kernel_size)
        self.conv2 = CircularConv3d(channels, channels, kernel_size)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return x + residual

class SubPixelDecoder3D(nn.Module):

    def __init__(self, embed_dim, out_channels, temporal_dim, spatial_dim, temporal_patch, spatial_patch, feat_dim=256, min_feat_dim=64):
        super().__init__()
        self.temporal_patch = temporal_patch
        self.spatial_patch = spatial_patch
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.num_t = temporal_dim // temporal_patch
        self.num_h = spatial_dim // spatial_patch
        self.num_w = spatial_dim // spatial_patch
        self.to_features = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, feat_dim), nn.GELU(), nn.Linear(feat_dim, feat_dim), nn.GELU())
        self.pre_mixing = nn.Sequential(CircularConv3d(feat_dim, feat_dim, kernel_size=5), nn.GELU(), ResBlock3d(feat_dim, kernel_size=5), ResBlock3d(feat_dim, kernel_size=3))
        self.upsample_stages = nn.ModuleList()
        current_dim = feat_dim
        if temporal_patch != 4 or spatial_patch != 4:
            raise ValueError('The released models use 4x4x4 patches.')
        for stage in range(2):
            shuffle_factor = 8
            shuffle_module = PixelShuffle3D(2)
            next_dim = max(current_dim // 2, min_feat_dim)
            stage_layers = nn.Sequential(CircularConv3d(current_dim, next_dim * shuffle_factor, kernel_size=3), nn.GELU(), shuffle_module, ResBlock3d(next_dim, kernel_size=3))
            self.upsample_stages.append(stage_layers)
            current_dim = next_dim
        self.final = nn.Sequential(ResBlock3d(current_dim, kernel_size=3), CircularConv3d(current_dim, current_dim, kernel_size=3), nn.GELU(), CircularConv3d(current_dim, out_channels, kernel_size=3))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if hasattr(self.final[-1], 'conv'):
            nn.init.xavier_uniform_(self.final[-1].conv.weight, gain=0.01)

    def forward(self, tokens):
        B = tokens.shape[0]
        tokens_flat = rearrange(tokens, 'b t (h w) d -> b (t h w) d', h=self.num_h, w=self.num_w)
        features = self.to_features(tokens_flat)
        features = rearrange(features, 'b (t h w) d -> b d t h w', t=self.num_t, h=self.num_h, w=self.num_w)
        features = self.pre_mixing(features)
        for stage in self.upsample_stages:
            features = stage(features)
        output = self.final(features)
        return output

class PreNormResidual(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x

class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):

    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else dim
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, context=None, causal_mask=False):
        context = x if context is None else context
        q = self.to_q(x)
        (k, v) = self.to_kv(context).chunk(2, dim=-1)
        (q, k, v) = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if causal_mask:
            mask = torch.ones(x.size(1), context.size(1), device=x.device, dtype=torch.bool).tril(diagonal=context.size(1) - x.size(1))
            dots = dots.masked_fill(~mask, float('-inf'))
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SpacetimeTransformerBlock(nn.Module):

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout, is_causal=False, has_cross_attention=False):
        super().__init__()
        self.is_causal = is_causal
        self.has_cross_attention = has_cross_attention
        self.spatial_self_attn = PreNormResidual(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.temporal_self_attn = PreNormResidual(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        if has_cross_attention:
            self.spatial_cross_attn = PreNormResidual(dim, Attention(dim, context_dim=dim, heads=heads, dim_head=dim_head, dropout=dropout))
            self.temporal_cross_attn = PreNormResidual(dim, Attention(dim, context_dim=dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNormResidual(dim, FeedForward(dim, mlp_dim, dropout))

    def forward(self, x, context=None):
        (B, T_patches, S_patches, D) = x.shape
        x_flat = rearrange(x, 'b t s d -> b (t s) d')
        x_spatial = rearrange(x_flat, 'b (t s) d -> (b t) s d', t=T_patches)
        x_spatial = self.spatial_self_attn(x_spatial)
        x_self_attd = rearrange(x_spatial, '(b t) s d -> b (t s) d', t=T_patches)
        x_temporal = rearrange(x_self_attd, 'b (t s) d -> (b s) t d', s=S_patches)
        x_temporal = self.temporal_self_attn(x_temporal, causal_mask=self.is_causal)
        x = rearrange(x_temporal, '(b s) t d -> b (t s) d', b=B)
        if self.has_cross_attention and context is not None:
            (context_t_patches, context_s_patches) = (context.shape[1], context.shape[2])
            context_flat = rearrange(context, 'b t s d -> b (t s) d')
            x_cross_s = rearrange(x, 'b (t s) d -> (b t) s d', t=T_patches)
            context_cross_s = rearrange(context_flat, 'b (t s) d -> (b t) s d', t=context_t_patches)
            x_cross_s = self.spatial_cross_attn(x_cross_s, context=context_cross_s)
            x = rearrange(x_cross_s, '(b t) s d -> b (t s) d', t=T_patches)
            x_cross_t = rearrange(x, 'b (t s) d -> (b s) t d', s=S_patches)
            context_cross_t = rearrange(context_flat, 'b (t s) d -> (b s) t d', s=context_s_patches)
            x_cross_t = self.temporal_cross_attn(x_cross_t, context=context_cross_t)
            x = rearrange(x_cross_t, '(b s) t d -> b (t s) d', b=B)
        x = self.ff(x)
        return rearrange(x, 'b (t s) d -> b t s d', t=T_patches)

class OverlappingPatchTokenizer(nn.Module):

    def __init__(self, in_channels, embed_dim, t_dim, s_dim, t_patch, s_patch):
        super().__init__()
        self.num_t_patches = t_dim // t_patch
        num_s_patches_side = s_dim // s_patch
        self.num_s_patches = num_s_patches_side ** 2
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=(t_patch, s_patch + 2, s_patch + 2), stride=(t_patch, s_patch, s_patch), padding=(0, 1, 1))
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed_t = nn.Parameter(torch.randn(1, self.num_t_patches, 1, 1, embed_dim) * 0.02)
        self.pos_embed_h = nn.Parameter(torch.randn(1, 1, num_s_patches_side, 1, embed_dim) * 0.02)
        self.pos_embed_w = nn.Parameter(torch.randn(1, 1, 1, num_s_patches_side, embed_dim) * 0.02)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b d t h w -> b t (h w) d')
        x = self.norm(x)
        pos = self.pos_embed_t + self.pos_embed_h + self.pos_embed_w
        pos = rearrange(pos, 'b t h w d -> b t (h w) d')
        return x + pos

class ViViViT(nn.Module):

    def __init__(self, config):
        super().__init__()
        embed_dim = config['embed_dim']
        encoder_depth = config['encoder_depth']
        decoder_depth = config['decoder_depth']
        heads = config['heads']
        dim_head = config.get('dim_head', embed_dim // heads)
        mlp_dim = config['mlp_dim']
        dropout = config.get('dropout', 0.1)
        in_channels = config.get('in_channels', 1)
        out_channels = config.get('out_channels', 1)
        temporal_dim = config['GridSize'][0]
        spatial_dim = config['GridSize'][1]
        t_patch = config['temporal_patch_size']
        s_patch = config['spatial_patch_size']
        self.use_checkpointing = config.get('use_checkpointing', True)
        num_t_patches = temporal_dim // t_patch
        num_s_patches_h = spatial_dim // s_patch
        num_s_patches_w = spatial_dim // s_patch
        num_s_patches_total = num_s_patches_h * num_s_patches_w
        self.encoder_tokenizer = OverlappingPatchTokenizer(in_channels, embed_dim, temporal_dim, spatial_dim, t_patch, s_patch)
        self.encoder_blocks = nn.ModuleList([SpacetimeTransformerBlock(embed_dim, heads, dim_head, mlp_dim, dropout, is_causal=True) for _ in range(encoder_depth)])
        self.decoder_init = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))
        self.decoder_blocks = nn.ModuleList([SpacetimeTransformerBlock(embed_dim, heads, dim_head, mlp_dim, dropout, is_causal=True, has_cross_attention=True) for _ in range(decoder_depth)])
        self.token_decoder = SubPixelDecoder3D(embed_dim=embed_dim, out_channels=out_channels, temporal_dim=temporal_dim, spatial_dim=spatial_dim, temporal_patch=t_patch, spatial_patch=s_patch, feat_dim=256, min_feat_dim=64)

    def _checkpoint(self, module, *args, **kwargs):
        if self.training and self.use_checkpointing:
            return checkpoint.checkpoint(module, *args, **kwargs, use_reentrant=False)
        else:
            return module(*args, **kwargs)

    def forward(self, x):
        (b, *_) = x.shape
        encoded_tokens = self.encoder_tokenizer(x)
        for block in self.encoder_blocks:
            encoded_tokens = self._checkpoint(block, encoded_tokens)
        decoded_tokens = self.decoder_init(encoded_tokens)
        for block in self.decoder_blocks:
            decoded_tokens = self._checkpoint(block, decoded_tokens, context=encoded_tokens)
        return self.token_decoder(decoded_tokens)
