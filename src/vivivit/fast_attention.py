from torch import nn
from torch.nn import functional as F
from einops import rearrange

from .transformer import PreNormResidual, FeedForward

class FastAttention(nn.Module):

    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else dim
        self.heads = heads
        self.dropout = dropout
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, context=None, causal_mask=False):
        context = x if context is None else context
        q = self.to_q(x)
        (k, v) = self.to_kv(context).chunk(2, dim=-1)
        (q, k, v) = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal_mask, dropout_p=self.dropout if self.training else 0.0)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FastSpacetimeTransformerBlock(nn.Module):

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout, is_causal=False, has_cross_attention=False):
        super().__init__()
        self.is_causal = is_causal
        self.has_cross_attention = has_cross_attention
        self.spatial_self_attn = PreNormResidual(dim, FastAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.temporal_self_attn = PreNormResidual(dim, FastAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        if has_cross_attention:
            self.spatial_cross_attn = PreNormResidual(dim, FastAttention(dim, context_dim=dim, heads=heads, dim_head=dim_head, dropout=dropout))
            self.temporal_cross_attn = PreNormResidual(dim, FastAttention(dim, context_dim=dim, heads=heads, dim_head=dim_head, dropout=dropout))
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
