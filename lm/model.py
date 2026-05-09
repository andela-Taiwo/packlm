"""The LM: a stack of pre-norm blocks, each a sequence mixer followed by an MLP.

Mixers are causal self-attention, a short causal depthwise convolution with a
gate, or a gated linear recurrence ("scan"). The MLP is dense or a mixture of
experts with a softmax router. Output embeddings are tied to the input ones.
Everything runs in float64.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

DTYPE = torch.float64


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d, dtype=DTYPE))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope(x, pos):
    """Rotate pairs of channels of x [B, H, T, Dh] by angles set by pos [B, T]."""
    dh = x.shape[-1]
    inv = 1.0 / (10000.0 ** (torch.arange(0, dh, 2, dtype=DTYPE) / dh))
    ang = pos.to(DTYPE)[:, None, :, None] * inv
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False, dtype=DTYPE)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False, dtype=DTYPE)
        self.use_rope = cfg.positions == "rope"

    def forward(self, x, pos):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        if self.use_rope:
            q, k = rope(q, pos), rope(k, pos)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.h)
        causal = torch.ones(t, t, dtype=torch.bool).tril()
        att = att.masked_fill(~causal, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, d)
        return self.proj(y)


class ShortConv(nn.Module):
    """y_t = sum_j w_j * x_{t-j} over the last `kernel` positions, gated by x_t."""

    def __init__(self, cfg):
        super().__init__()
        self.k = cfg.conv_kernel
        self.weight = nn.Parameter(torch.empty(cfg.d_model, self.k, dtype=DTYPE))
        self.gate = nn.Linear(cfg.d_model, cfg.d_model, bias=False, dtype=DTYPE)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False, dtype=DTYPE)

    def forward(self, x, pos):
        b, t, d = x.shape
        xp = F.pad(x, (0, 0, self.k - 1, 0))
        y = torch.zeros_like(x)
        for j in range(self.k):
            y = y + xp[:, self.k - 1 - j: self.k - 1 - j + t] * self.weight[:, j]
        return self.proj(y * F.silu(self.gate(x)))


class Scan(nn.Module):
    """h_t = a_t * h_{t-1} + (1 - a_t) * u_t, with a_t and u_t computed from x_t."""

    def __init__(self, cfg):
        super().__init__()
        self.a = nn.Linear(cfg.d_model, cfg.d_model, dtype=DTYPE)
        self.u = nn.Linear(cfg.d_model, cfg.d_model, bias=False, dtype=DTYPE)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False, dtype=DTYPE)

    def forward(self, x, pos):
        a = torch.sigmoid(self.a(x))
        u = self.u(x)
        h = torch.zeros_like(x[:, 0])
        out = []
        for i in range(x.shape[1]):
            h = a[:, i] * h + (1 - a[:, i]) * u[:, i]
            out.append(h)
        return self.proj(torch.stack(out, 1))


class MLP(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.up = nn.Linear(d, d_ff, bias=False, dtype=DTYPE)
        self.down = nn.Linear(d_ff, d, bias=False, dtype=DTYPE)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class MoE(nn.Module):
    """Top-k routing over `experts` MLPs; the chosen experts' outputs are mixed by
    their router probabilities, renormalised over the k chosen.

    With a capacity factor c > 0 each expert takes at most ceil(c * k * N / E)
    of the N tokens' assignments, first come first served; an assignment over
    capacity is dropped and that expert adds nothing for that token."""

    def __init__(self, cfg):
        super().__init__()
        self.k = cfg.top_k
        self.capacity_factor = cfg.capacity_factor
        self.router = nn.Linear(cfg.d_model, cfg.experts, bias=False, dtype=DTYPE)
        self.experts = nn.ModuleList(MLP(cfg.d_model, cfg.d_ff) for _ in range(cfg.experts))

    def forward(self, x):
        probs = self.router(x).softmax(-1)
        top_p, top_i = probs.topk(self.k, dim=-1)
        top_p = top_p / top_p.sum(-1, keepdim=True)
        keep = torch.ones_like(top_i, dtype=torch.bool)
        if self.capacity_factor > 0:
            n_exp = probs.shape[-1]
            cap = math.ceil(self.capacity_factor * self.k * top_i[..., 0].numel() / n_exp)
            flat_i, flat_keep = top_i.reshape(-1), keep.reshape(-1)
            for e in range(n_exp):
                taken = (flat_i == e).nonzero().squeeze(-1)
                flat_keep[taken[cap:]] = False
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            w = (top_p * ((top_i == e) & keep)).sum(-1, keepdim=True)
            sel = w.squeeze(-1) > 0
            if sel.any():
                out[sel] = out[sel] + w[sel] * expert(x[sel])
        return out, (probs, top_i)


MIXER = {"attn": Attention, "conv": ShortConv, "scan": Scan}


class Block(nn.Module):
    def __init__(self, cfg, mixer):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = MIXER[mixer](cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.moe = cfg.mlp == "moe"
        self.mlp = MoE(cfg) if self.moe else MLP(cfg.d_model, cfg.d_ff)

    def forward(self, x, pos):
        x = x + self.mixer(self.norm1(x), pos)
        if self.moe:
            y, route = self.mlp(self.norm2(x))
            return x + y, route
        return x + self.mlp(self.norm2(x)), None


class LM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model, dtype=DTYPE)
        if cfg.positions == "learned":
            self.pos = nn.Embedding(cfg.ctx, cfg.d_model, dtype=DTYPE)
        self.blocks = nn.ModuleList(Block(cfg, m) for m in cfg.mixers)
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, tokens):
        """tokens [B, T] -> logits [B, T, vocab] and, per MoE layer, (probs, top_i)."""
        b, t = tokens.shape
        pos = torch.arange(t).expand(b, t)
        x = self.embed(tokens)
        if self.cfg.positions == "learned":
            x = x + self.pos(pos)
        routes = []
        for blk in self.blocks:
            x, route = blk(x, pos)
            if route is not None:
                routes.append(route)
        return self.norm(x) @ self.embed.weight.T, routes


def init_weights(model, seed):
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        with torch.no_grad():
            if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name == "norm.weight":
                p.fill_(1.0)
            elif name.endswith(".bias"):
                p.zero_()
            else:
                fan_in = p.shape[-1] if p.dim() > 1 else p.shape[0]
                p.copy_(torch.randn(p.shape, generator=g, dtype=DTYPE) / math.sqrt(fan_in))
