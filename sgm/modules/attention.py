import logging
import math
from inspect import isfunction
from typing import Any, Optional
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version
from torch import nn
from .diffusionmodules.util import checkpoint
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from ..modules.diffusionmodules.util import zero_module
from ..modules.nerfsd_pytorch3d import NerfSDModule, VolRender

logpy = logging.getLogger(__name__)

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
    logpy.warn(
        f"No SDP backend available, likely because you are running in pytorch "
        f"versions < 2.0. In fact, you are using PyTorch {torch.__version__}. "
        f"You might want to consider upgrading."
    )

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    logpy.warn("no module 'xformers'. Processing without...")


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


trunc_exp = _TruncExp.apply
"""Same as torch.exp, but with the backward pass clipped to prevent vanishing/exploding
gradients."""


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp
            )

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        ## old
        """
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        """
        ## new
        with sdp_kernel(**BACKEND_MAP[self.backend]):
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, add_lora=False, **kwargs
    ):
        super().__init__()
        logpy.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, "
            f"context_dim is {context_dim} and using {heads} heads with a "
            f"dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head
        self.add_lora = add_lora

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        if add_lora:
            r = 32
            self.to_q_attn3_down = nn.Linear(query_dim, r, bias=False)
            self.to_q_attn3_up = zero_module(nn.Linear(r, inner_dim, bias=False))
            self.to_k_attn3_down = nn.Linear(context_dim, r, bias=False)
            self.to_k_attn3_up = zero_module(nn.Linear(r, inner_dim, bias=False))
            self.to_v_attn3_down = nn.Linear(context_dim, r, bias=False)
            self.to_v_attn3_up = zero_module(nn.Linear(r, inner_dim, bias=False))
            self.to_o_attn3_down = nn.Linear(inner_dim, r, bias=False)
            self.to_o_attn3_up = zero_module(nn.Linear(r, query_dim, bias=False))
            self.dropoutq = nn.Dropout(0.1)
            self.dropoutk = nn.Dropout(0.1)
            self.dropoutv = nn.Dropout(0.1)
            self.dropouto = nn.Dropout(0.1)

            nn.init.normal_(self.to_q_attn3_down.weight, std=1 / r)
            nn.init.normal_(self.to_k_attn3_down.weight, std=1 / r)
            nn.init.normal_(self.to_v_attn3_down.weight, std=1 / r)
            nn.init.normal_(self.to_o_attn3_down.weight, std=1 / r)

        self.attention_op: Optional[Any] = None

    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        context_k = context  # b, n, c, h, w

        q = self.to_q(x)
        context = default(context, x)
        context_k = default(context_k, x)
        k = self.to_k(context_k)
        v = self.to_v(context_k)
        if self.add_lora:
            q += self.dropoutq(self.to_q_attn3_up(self.to_q_attn3_down(x)))
            k += self.dropoutk(self.to_k_attn3_up(self.to_k_attn3_down(context_k)))
            v += self.dropoutv(self.to_v_attn3_up(self.to_v_attn3_down(context_k)))

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        attn_bias = None

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=attn_bias, op=self.attention_op
        )

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        final = self.to_out(out)
        if self.add_lora:
            final += self.dropouto(self.to_o_attn3_up(self.to_o_attn3_down(out)))
        return final


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attn_mode="softmax",
        sdp_backend=None,
        image_cross=False,
        far=2,
        num_samples=32,
        add_lora=False,
        rgb_predict=False,
        mode='pixel-nerf',
        average=False,
        num_freqs=16,
        use_prev_weights_imp_sample=False,
        imp_sample_next_step=False,
        stratified=False,
        imp_sampling_percent=0.9,
        near_plane=0.
    ):

        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        self.add_lora = add_lora
        self.image_cross = image_cross
        self.rgb_predict = rgb_predict
        self.use_prev_weights_imp_sample = use_prev_weights_imp_sample
        self.imp_sample_next_step = imp_sample_next_step
        self.rendered_feat = None
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            logpy.warn(
                f"Attention mode '{attn_mode}' is not available. Falling "
                f"back to native attention. This is not a problem in "
                f"Pytorch >= 2.0. FYI, you are running with PyTorch "
                f"version {torch.__version__}."
            )
            attn_mode = "softmax"
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            logpy.warn(
                "We do not support vanilla attention anymore, as it is too "
                "expensive. Sorry."
            )
            if not XFORMERS_IS_AVAILABLE:
                assert (
                    False
                ), "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                logpy.info("Falling back to xformers efficient attention.")
                attn_mode = "softmax-xformers"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            add_lora=self.add_lora,
            context_dim=context_dim if self.disable_self_attn else None,
            backend=sdp_backend,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            add_lora=self.add_lora,
            backend=sdp_backend,
        )  # is self-attn if context is none
        if image_cross:
            self.pose_emb_layers = nn.Linear(2*dim, dim, bias=False)
            nn.init.eye_(self.pose_emb_layers.weight)
            self.pose_featurenerf = NerfSDModule(mode=mode,
                                                 out_channels=dim,
                                                 far_plane=far,
                                                 num_samples=num_samples,
                                                 rgb_predict=rgb_predict,
                                                 average=average,
                                                 num_freqs=num_freqs,
                                                 stratified=stratified,
                                                 imp_sampling_percent=imp_sampling_percent,
                                                 near_plane=near_plane,
                                                 )

            self.renderer = VolRender()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        if self.checkpoint:
            logpy.debug(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x, context=None, context_ref=None, pose=None, mask_ref=None, prev_weights=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        kwargs = {"x": x}

        if context is not None:
            kwargs.update({"context": context})

        if context_ref is not None:
            kwargs.update({"context_ref": context_ref})

        if pose is not None:
            kwargs.update({"pose": pose})

        if mask_ref is not None:
            kwargs.update({"mask_ref": mask_ref})

        if prev_weights is not None:
            kwargs.update({"prev_weights": prev_weights})

        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})

        if n_times_crossframe_attn_in_self:
            kwargs.update(
                {"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self}
            )

        # return mixed_checkpoint(self._forward, kwargs, self.parameters(), self.checkpoint)
        return checkpoint(
            self._forward, (x, context, context_ref, pose, mask_ref, prev_weights), self.parameters(), self.checkpoint
        )

    def reference_attn(self, x, context_ref, context, pose, prev_weights, mask_ref):
        feats, sigmas, dists, _, predicted_rgb, sigmas_uniform, dists_uniform = self.pose_featurenerf(pose,
                                                                                                      context_ref,
                                                                                                      mask_ref,
                                                                                                      prev_weights=prev_weights if self.use_prev_weights_imp_sample else None,
                                                                                                      imp_sample_next_step=self.imp_sample_next_step)

        b, hw, d = feats.size()[:3]
        feats = rearrange(feats, "b hw d ... -> b (hw d) ...")

        feats = (
            self.attn2(
                self.norm2(feats), context=context,
            )
            + feats
        )

        feats = rearrange(feats, "b (hw d) ... -> b hw d ...", hw=hw, d=d)

        sigmas_ = trunc_exp(sigmas)
        if sigmas_uniform is not None:
            sigmas_uniform = trunc_exp(sigmas_uniform)

        context_ref, fg_mask, alphas, weights_uniform, predicted_rgb = self.renderer(feats, sigmas_, dists, densities_uniform=sigmas_uniform, dists_uniform=dists_uniform, return_weights_uniform=True, rgb=F.sigmoid(predicted_rgb) if predicted_rgb is not None else None)
        if self.use_prev_weights_imp_sample:
            prev_weights = weights_uniform

        return context_ref, fg_mask, prev_weights, alphas, predicted_rgb

    def _forward(
        self, x, context=None, context_ref=None, pose=None, mask_ref=None, prev_weights=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
    ):
        fg_mask = None
        weights = None
        alphas = None
        predicted_rgb = None
        xref = None

        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                if not self.disable_self_attn
                else 0,
            )
            + x
        )
        x = (
            self.attn2(
                self.norm2(x), context=context, additional_tokens=additional_tokens
            )
            + x
        )
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            if context_ref is not None:
                xref, fg_mask, weights, alphas, predicted_rgb = self.reference_attn(x,
                                                                                    rearrange(context_ref, "(b n) ... -> b n ...", b=x.size(0), n=context_ref.size(0) // x.size(0)),
                                                                                    context,
                                                                                    pose,
                                                                                    prev_weights,
                                                                                    mask_ref)
                x = self.pose_emb_layers(torch.cat([x, xref], -1))

        x = self.ff(self.norm3(x)) + x
        return x, fg_mask, weights, alphas, predicted_rgb


class BasicTransformerSingleLayerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention  # on the A100s not quite as fast as the above version
        # (todo might depend on head_dim, check, falls back to semi-optimized kernels for dim!=[16,32,64,128])
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        attn_mode="softmax",
    ):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context) + x
        x = self.ff(self.norm2(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax",
        use_checkpoint=True,
        # sdp_backend=SDPBackend.FLASH_ATTENTION
        sdp_backend=None,
        image_cross=True,
        rgb_predict=False,
        far=2,
        num_samples=32,
        add_lora=False,
        mode='feature-nerf',
        average=False,
        num_freqs=16,
        use_prev_weights_imp_sample=False,
        stratified=False,
        poscontrol_interval=4,
        imp_sampling_percent=0.9,
        near_plane=0.
    ):
        super().__init__()
        logpy.debug(
            f"constructing {self.__class__.__name__} of depth {depth} w/ "
            f"{in_channels} channels and {n_heads} heads."
        )
        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                logpy.warn(
                    f"{self.__class__.__name__}: Found context dims "
                    f"{context_dim} of depth {len(context_dim)}, which does not "
                    f"match the specified 'depth' of {depth}. Setting context_dim "
                    f"to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.image_cross = image_cross
        self.poscontrol_interval = poscontrol_interval

        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                    image_cross=self.image_cross and (d % poscontrol_interval == 0),
                    far=far,
                    num_samples=num_samples,
                    add_lora=add_lora and self.image_cross and (d % poscontrol_interval == 0),
                    rgb_predict=rgb_predict,
                    mode=mode,
                    average=average,
                    num_freqs=num_freqs,
                    use_prev_weights_imp_sample=use_prev_weights_imp_sample,
                    imp_sample_next_step=(use_prev_weights_imp_sample and self.image_cross and (d % poscontrol_interval == 0) and depth >= poscontrol_interval and d < (depth // poscontrol_interval) * poscontrol_interval),
                    stratified=stratified,
                    imp_sampling_percent=imp_sampling_percent,
                    near_plane=near_plane,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, xr, context=None, contextr=None, pose=None, mask_ref=None, prev_weights=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if xr is None:
            if not isinstance(context, list):
                context = [context]
            b, c, h, w = x.shape
            x_in = x
            x = self.norm(x)
            if not self.use_linear:
                x = self.proj_in(x)
            x = rearrange(x, "b c h w -> b (h w) c").contiguous()
            if self.use_linear:
                x = self.proj_in(x)
            for i, block in enumerate(self.transformer_blocks):
                if i > 0 and len(context) == 1:
                    i = 0  # use same context for each block
                x, _, _, _, _ = block(x, context=context[i])
            if self.use_linear:
                x = self.proj_out(x)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
            if not self.use_linear:
                x = self.proj_out(x)
            return x + x_in, None, None, None, None, None
        else:
            if not isinstance(context, list):
                context = [context]
                contextr = [contextr]
            b, c, h, w = x.shape
            b1, _, _, _ = xr.shape
            x_in = x
            xr_in = xr
            fg_masks = []
            alphas = []
            rgbs = []

            x = self.norm(x)
            with torch.no_grad():
                xr = self.norm(xr)

            if not self.use_linear:
                x = self.proj_in(x)
                with torch.no_grad():
                    xr = self.proj_in(xr)

            x = rearrange(x, "b c h w -> b (h w) c").contiguous()
            xr = rearrange(xr, "b1 c h w -> b1 (h w) c").contiguous()
            if self.use_linear:
                x = self.proj_in(x)
                with torch.no_grad():
                    xr = self.proj_in(xr)

            prev_weights = None
            counter = 0
            for i, block in enumerate(self.transformer_blocks):
                if i > 0 and len(context) == 1:
                    i = 0  # use same context for each block
                if self.image_cross and (counter % self.poscontrol_interval == 0):
                    with torch.no_grad():
                        xr, _, _, _, _ = block(xr, context=contextr[i])
                    x, fg_mask, weights, alpha, rgb = block(x, context=context[i], context_ref=xr.detach(), pose=pose, mask_ref=mask_ref, prev_weights=prev_weights)
                    prev_weights = weights
                    fg_masks.append(fg_mask)
                    if alpha is not None:
                        alphas.append(alpha)
                    if rgb is not None:
                        rgbs.append(rgb)
                else:
                    with torch.no_grad():
                        xr, _, _, _, _ = block(xr, context=contextr[i])
                    x, _, _, _, _ = block(x, context=context[i])
                counter += 1
            if self.use_linear:
                x = self.proj_out(x)
                with torch.no_grad():
                    xr = self.proj_out(xr)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
            xr = rearrange(xr, "b1 (h w) c -> b1 c h w", h=h, w=w).contiguous()
            if not self.use_linear:
                x = self.proj_out(x)
                with torch.no_grad():
                    xr = self.proj_out(xr)
            if len(fg_masks) > 0:
                if len(rgbs) <= 0:
                    rgbs = None
                if len(alphas) <= 0:
                    alphas = None
                return x + x_in, (xr + xr_in).detach(), fg_masks, prev_weights, alphas, rgbs
            else:
                return x + x_in, (xr + xr_in).detach(), None, prev_weights, None, None


def benchmark_attn():
    # Lets define a helpful benchmarking function:
    # https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import torch.nn.functional as F
    import torch.utils.benchmark as benchmark

    def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
        t0 = benchmark.Timer(
            stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
        )
        return t0.blocked_autorange().mean * 1e6

    # Lets define the hyper-parameters of our input
    batch_size = 32
    max_sequence_len = 1024
    num_heads = 32
    embed_dimension = 32

    dtype = torch.float16

    query = torch.rand(
        batch_size,
        num_heads,
        max_sequence_len,
        embed_dimension,
        device=device,
        dtype=dtype,
    )
    key = torch.rand(
        batch_size,
        num_heads,
        max_sequence_len,
        embed_dimension,
        device=device,
        dtype=dtype,
    )
    value = torch.rand(
        batch_size,
        num_heads,
        max_sequence_len,
        embed_dimension,
        device=device,
        dtype=dtype,
    )

    print(f"q/k/v shape:", query.shape, key.shape, value.shape)

    # Lets explore the speed of each of the 3 implementations
    from torch.backends.cuda import SDPBackend, sdp_kernel

    # Helpful arguments mapper
    backend_map = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
    }

    from torch.profiler import ProfilerActivity, profile, record_function

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    print(
        f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds"
    )
    with profile(
        activities=activities, record_shapes=False, profile_memory=True
    ) as prof:
        with record_function("Default detailed stats"):
            for _ in range(25):
                o = F.scaled_dot_product_attention(query, key, value)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print(
        f"The math implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds"
    )
    with sdp_kernel(**backend_map[SDPBackend.MATH]):
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("Math implmentation stats"):
                for _ in range(25):
                    o = F.scaled_dot_product_attention(query, key, value)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
        try:
            print(
                f"The flash attention implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds"
            )
        except RuntimeError:
            print("FlashAttention is not supported. See warnings for reasons.")
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("FlashAttention stats"):
                for _ in range(25):
                    o = F.scaled_dot_product_attention(query, key, value)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
        try:
            print(
                f"The memory efficient implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds"
            )
        except RuntimeError:
            print("EfficientAttention is not supported. See warnings for reasons.")
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("EfficientAttention stats"):
                for _ in range(25):
                    o = F.scaled_dot_product_attention(query, key, value)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def run_model(model, x, context):
    return model(x, context)


def benchmark_transformer_blocks():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import torch.utils.benchmark as benchmark

    def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
        t0 = benchmark.Timer(
            stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
        )
        return t0.blocked_autorange().mean * 1e6

    checkpoint = True
    compile = False

    batch_size = 32
    h, w = 64, 64
    context_len = 77
    embed_dimension = 1024
    context_dim = 1024
    d_head = 64

    transformer_depth = 4

    n_heads = embed_dimension // d_head

    dtype = torch.float16

    model_native = SpatialTransformer(
        embed_dimension,
        n_heads,
        d_head,
        context_dim=context_dim,
        use_linear=True,
        use_checkpoint=checkpoint,
        attn_type="softmax",
        depth=transformer_depth,
        sdp_backend=SDPBackend.FLASH_ATTENTION,
    ).to(device)
    model_efficient_attn = SpatialTransformer(
        embed_dimension,
        n_heads,
        d_head,
        context_dim=context_dim,
        use_linear=True,
        depth=transformer_depth,
        use_checkpoint=checkpoint,
        attn_type="softmax-xformers",
    ).to(device)
    if not checkpoint and compile:
        print("compiling models")
        model_native = torch.compile(model_native)
        model_efficient_attn = torch.compile(model_efficient_attn)

    x = torch.rand(batch_size, embed_dimension, h, w, device=device, dtype=dtype)
    c = torch.rand(batch_size, context_len, context_dim, device=device, dtype=dtype)

    from torch.profiler import ProfilerActivity, profile, record_function

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with torch.autocast("cuda"):
        print(
            f"The native model runs in {benchmark_torch_function_in_microseconds(model_native.forward, x, c):.3f} microseconds"
        )
        print(
            f"The efficientattn model runs in {benchmark_torch_function_in_microseconds(model_efficient_attn.forward, x, c):.3f} microseconds"
        )

        print(75 * "+")
        print("NATIVE")
        print(75 * "+")
        torch.cuda.reset_peak_memory_stats()
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("NativeAttention stats"):
                for _ in range(25):
                    model_native(x, c)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(torch.cuda.max_memory_allocated() * 1e-9, "GB used by native block")

        print(75 * "+")
        print("Xformers")
        print(75 * "+")
        torch.cuda.reset_peak_memory_stats()
        with profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            with record_function("xformers stats"):
                for _ in range(25):
                    model_efficient_attn(x, c)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print(torch.cuda.max_memory_allocated() * 1e-9, "GB used by xformers block")


def test01():
    # conv1x1 vs linear
    from ..util import count_params

    conv = nn.Conv2d(3, 32, kernel_size=1).cuda()
    print(count_params(conv))
    linear = torch.nn.Linear(3, 32).cuda()
    print(count_params(linear))

    print(conv.weight.shape)

    # use same initialization
    linear.weight = torch.nn.Parameter(conv.weight.squeeze(-1).squeeze(-1))
    linear.bias = torch.nn.Parameter(conv.bias)

    print(linear.weight.shape)

    x = torch.randn(11, 3, 64, 64).cuda()

    xr = rearrange(x, "b c h w -> b (h w) c").contiguous()
    print(xr.shape)
    out_linear = linear(xr)
    print(out_linear.mean(), out_linear.shape)

    out_conv = conv(x)
    print(out_conv.mean(), out_conv.shape)
    print("done with test01.\n")


def test02():
    # try cosine flash attention
    import time

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("testing cosine flash attention...")
    DIM = 1024
    SEQLEN = 4096
    BS = 16

    print(" softmax (vanilla) first...")
    model = BasicTransformerBlock(
        dim=DIM,
        n_heads=16,
        d_head=64,
        dropout=0.0,
        context_dim=None,
        attn_mode="softmax",
    ).cuda()
    try:
        x = torch.randn(BS, SEQLEN, DIM).cuda()
        tic = time.time()
        y = model(x)
        toc = time.time()
        print(y.shape, toc - tic)
    except RuntimeError as e:
        # likely oom
        print(str(e))

    print("\n now flash-cosine...")
    model = BasicTransformerBlock(
        dim=DIM,
        n_heads=16,
        d_head=64,
        dropout=0.0,
        context_dim=None,
        attn_mode="flash-cosine",
    ).cuda()
    x = torch.randn(BS, SEQLEN, DIM).cuda()
    tic = time.time()
    y = model(x)
    toc = time.time()
    print(y.shape, toc - tic)
    print("done with test02.\n")


if __name__ == "__main__":
    # test01()
    # test02()
    # test03()

    # benchmark_attn()
    benchmark_transformer_blocks()

    print("done.")
