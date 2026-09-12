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
