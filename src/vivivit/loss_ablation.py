import torch
from torch import nn
import torch.utils.checkpoint as checkpoint

from .transformer import OverlappingPatchTokenizer, SubPixelDecoder3D
from .fast_attention import FastSpacetimeTransformerBlock

class LossAblationViViT(nn.Module):

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
        self.encoder_blocks = nn.ModuleList([FastSpacetimeTransformerBlock(embed_dim, heads, dim_head, mlp_dim, dropout, is_causal=True) for _ in range(encoder_depth)])
        self.decoder_queries = nn.Parameter(torch.randn(1, num_t_patches, num_s_patches_total, embed_dim) * .02)
        self.decoder_blocks = nn.ModuleList([FastSpacetimeTransformerBlock(embed_dim, heads, dim_head, mlp_dim, dropout, is_causal=True, has_cross_attention=True) for i in range(decoder_depth)])
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
        decoded_tokens = self.decoder_queries.expand(b, -1, -1, -1)
        for block in self.decoder_blocks:
            decoded_tokens = self._checkpoint(block, decoded_tokens, context=encoded_tokens)
        return self.token_decoder(decoded_tokens)


def build_model(config):
    return LossAblationViViT(config)
