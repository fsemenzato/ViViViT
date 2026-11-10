import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import torch.utils.checkpoint as checkpoint

from .transformer import PixelShuffle3D
from .fast_attention import FastSpacetimeTransformerBlock

class CircularConv3d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=0)

    def forward(self, x):
        x = F.pad(x, [self.pad] * 6, mode='circular')
        if x.dim() == 5:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        return self.conv(x)
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
class SlimOverlappingPatchTokenizer(nn.Module):

    def __init__(self, in_channels, embed_dim, t_dim, s_dim, t_patch, s_patch):
        super().__init__()
        self.num_t_patches = t_dim // t_patch
        num_s_patches_side = s_dim // s_patch
        self.num_s_patches = num_s_patches_side ** 2
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=(t_patch, s_patch + 2, s_patch + 2), stride=(t_patch, s_patch, s_patch), padding=0)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed_t = nn.Parameter(torch.randn(1, self.num_t_patches, 1, 1, embed_dim) * 0.02)
        self.pos_embed_h = nn.Parameter(torch.randn(1, 1, num_s_patches_side, 1, embed_dim) * 0.02)
        self.pos_embed_w = nn.Parameter(torch.randn(1, 1, 1, num_s_patches_side, embed_dim) * 0.02)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1, 0, 0), mode='circular')
        x = self.proj(x)
        x = rearrange(x, 'b d t h w -> b t (h w) d')
        x = self.norm(x)
        pos = self.pos_embed_t + self.pos_embed_h + self.pos_embed_w
        pos = rearrange(pos, 'b t h w d -> b t (h w) d')
        return x + pos
class MemoryViViT(nn.Module):

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
        self.checkpoint_decoder_cnn = self.use_checkpointing
        feat_dim = 64
        min_feat_dim = 64
        encoder_is_causal = True
        decoder_is_causal = True
        self.encoder_is_causal = encoder_is_causal
        self.decoder_is_causal = decoder_is_causal
        num_t_patches = temporal_dim // t_patch
        num_s_patches_total = (spatial_dim // s_patch) ** 2
        self.encoder_tokenizer = SlimOverlappingPatchTokenizer(in_channels, embed_dim, temporal_dim, spatial_dim, t_patch, s_patch)
        self.encoder_blocks = nn.ModuleList([FastSpacetimeTransformerBlock(embed_dim, heads, dim_head, mlp_dim, dropout, is_causal=encoder_is_causal) for _ in range(encoder_depth)])
        self.decoder_init = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))
        self.decoder_blocks = nn.ModuleList([FastSpacetimeTransformerBlock(embed_dim, heads, dim_head, mlp_dim, dropout, is_causal=decoder_is_causal, has_cross_attention=True) for i in range(decoder_depth)])
        self.token_decoder = SubPixelDecoder3D(embed_dim=embed_dim, out_channels=out_channels, temporal_dim=temporal_dim, spatial_dim=spatial_dim, temporal_patch=t_patch, spatial_patch=s_patch, feat_dim=feat_dim, min_feat_dim=min_feat_dim)
        self.token_decoder = self.token_decoder.to(memory_format=torch.channels_last_3d)

    def _checkpoint(self, module, *args, **kwargs):
        if self.training and self.use_checkpointing:
            return checkpoint.checkpoint(module, *args, **kwargs, use_reentrant=False)
        return module(*args, **kwargs)

    def _decode(self, tokens):
        d = self.token_decoder
        if not (self.training and self.checkpoint_decoder_cnn):
            return d(tokens)
        B = tokens.shape[0]
        feats = rearrange(tokens, 'b t (h w) e -> b (t h w) e', h=d.num_h, w=d.num_w)
        feats = d.to_features(feats)
        feats = rearrange(feats, 'b (t h w) e -> b e t h w', t=d.num_t, h=d.num_h, w=d.num_w)
        feats = checkpoint.checkpoint(d.pre_mixing, feats, use_reentrant=False)
        for stage in d.upsample_stages:
            feats = checkpoint.checkpoint(stage, feats, use_reentrant=False)
        return checkpoint.checkpoint(d.final, feats, use_reentrant=False)

    def forward(self, x):
        (b, *_) = x.shape
        encoded_tokens = self.encoder_tokenizer(x)
        for block in self.encoder_blocks:
            encoded_tokens = self._checkpoint(block, encoded_tokens)
        decoded_tokens = self.decoder_init(encoded_tokens)
        for block in self.decoder_blocks:
            decoded_tokens = self._checkpoint(block, decoded_tokens, context=encoded_tokens)
        return self._decode(decoded_tokens)
class TimeLocalGroupNorm(nn.Module):

    def __init__(self, num_groups, C):
        super().__init__()
        g = num_groups
        if g > C:
            g = C
        while g > 1 and C % g != 0:
            g //= 2
        self.gn = nn.GroupNorm(g if g > 0 else 1, C)

    def forward(self, x):
        (N, C, T, H, W) = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)
        x = self.gn(x)
        return x.reshape(N, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

class TemporalConvBlock(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.norm1 = TimeLocalGroupNorm(8, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.norm2 = TimeLocalGroupNorm(8, cout)
        self.relu = nn.ReLU(inplace=True)
        self.res = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        idn = self.res(x)
        o = self.relu(self.norm1(self.conv1(x)))
        o = self.norm2(self.conv2(o))
        return self.relu(o + idn)

class UNet3D_TL(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, base_channels=32, num_levels=4, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        ch = [base_channels * 2 ** i for i in range(num_levels + 1)]
        self.init = TemporalConvBlock(in_channels, ch[0])
        self.enc = nn.ModuleList([TemporalConvBlock(ch[i], ch[i + 1]) for i in range(num_levels)])
        self.pool = nn.ModuleList([nn.MaxPool3d(2) for _ in range(num_levels)])
        self.bott = TemporalConvBlock(ch[num_levels], ch[num_levels])
        self.up = nn.ModuleList([nn.ConvTranspose3d(ch[i + 1], ch[i], 2, 2) for i in range(num_levels - 1, -1, -1)])
        self.dec = nn.ModuleList([TemporalConvBlock(ch[i] * 2, ch[i]) for i in range(num_levels - 1, -1, -1)])
        self.final = nn.Conv3d(ch[0], out_channels, 1)

    def _run(self, blk, x):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(blk, x, use_reentrant=False)
        return blk(x)

    def forward(self, x):
        skips = []
        o = self._run(self.init, x)
        for (blk, pl) in zip(self.enc, self.pool):
            skips.append(o)
            o = pl(self._run(blk, o))
        o = self._run(self.bott, o)
        for (i, (up, blk)) in enumerate(zip(self.up, self.dec)):
            o = up(o)
            s = skips[-(i + 1)]
            if o.shape != s.shape:
                o = F.interpolate(o, size=s.shape[2:], mode='trilinear', align_corners=False)
            o = self._run(blk, torch.cat([o, s], 1))
        return self.final(o)


def build_model(config):
    if config['model_type'] == 'vivit_overlap_slim':
        return MemoryViViT(config)
    if config['model_type'] == 'unet_tl':
        return UNet3D_TL(base_channels=config['base_channels'], num_levels=config['num_levels'], use_checkpoint=config['use_checkpointing'])
    raise ValueError(config['model_type'])
