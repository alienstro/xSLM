"""The hand-written modules of the xSLM model.

Every parameter carries the name that Llama gives it, so the state dictionary
matches LlamaForCausalLM key for key. The parity test asserts this.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel

from xslm.config import rope_theta_of


class RMSNorm(nn.Module):
    """Root mean square normalization, with a scale and no bias."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def build_rope_cache(head_dim, max_seq_len, theta, device, dtype):
    """Return the cosine table and the sine table for rotary position encoding."""
    exponent = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    inverse_frequency = 1.0 / (theta**exponent)
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inverse_frequency)
    table = torch.cat((angles, angles), dim=-1)
    return table.cos().to(dtype), table.sin().to(dtype)


def rotate_half(x):
    """Rotate the two halves of the last dimension. This is the Llama layout."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(q, k, cos, sin):
    """Rotate the queries and the keys. The values stay unchanged."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    """Multi-head causal attention. The kernel choice belongs to PyTorch."""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, cos, sin):
        batch, seq_len, hidden_size = hidden_states.shape
        shape = (batch, seq_len, self.num_heads, self.head_dim)
        q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:seq_len], sin[:seq_len])
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, seq_len, hidden_size)
        return self.o_proj(attended)


class MLP(nn.Module):
    """The SwiGLU feed-forward network, with no bias."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DecoderLayer(nn.Module):
    """One pre-normalization decoder block."""

    def __init__(self, hidden_size, num_heads, intermediate_size, eps=1e-5):
        super().__init__()
        self.self_attn = Attention(hidden_size, num_heads)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)

    def forward(self, hidden_states, cos, sin):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), cos, sin)
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class XSLMModel(LlamaPreTrainedModel):
    """The embedding, the stack of decoder layers, and the final norm."""

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.rms_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        head_dim = config.hidden_size // config.num_attention_heads
        cos, sin = build_rope_cache(
            head_dim,
            config.max_position_embeddings,
            rope_theta_of(config),
            torch.device("cpu"),
            torch.float32,
        )
        # persistent=False keeps the tables out of the state dictionary, so the
        # parity test against LlamaForCausalLM passes.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.post_init()

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        cos = self.rope_cos.to(hidden_states.dtype)
        sin = self.rope_sin.to(hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)
        return self.norm(hidden_states)


class XSLMForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    """The decoder with a tied output projection."""

    # transformers 5 reads this map to tie the output projection to the embedding.
    # Without it, post_init leaves two separate tensors, and the parameter count grows
    # by vocab_size times hidden_size.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = XSLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self._scale_the_residual_projections()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def _scale_the_residual_projections(self):
        """Shrink the projections that write into the residual stream.

        The smaller standard deviation keeps the variance of the residual stream
        stable as the depth increases. See section 3.2 of the design spec.
        """
        std = self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)
        for name, parameter in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=std)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, labels=None, **kwargs):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


def build_model(config):
    """Return one XSLMForCausalLM for the given configuration."""
    return XSLMForCausalLM(config)
