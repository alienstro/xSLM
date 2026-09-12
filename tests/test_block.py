import torch

from xslm.model import MLP, Attention, DecoderLayer, build_rope_cache


def small():
    return dict(hidden_size=64, num_heads=4, intermediate_size=128, eps=1e-5)


def test_no_module_holds_a_bias():
    layer = DecoderLayer(**small())
    names = [name for name, _ in layer.named_parameters()]
    assert not any(name.endswith("bias") for name in names)


def test_the_parameter_names_copy_the_llama_names():
    names = {name for name, _ in DecoderLayer(**small()).named_parameters()}
    assert names == {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    }


def test_the_block_keeps_the_shape():
    layer = DecoderLayer(**small())
    cos, sin = build_rope_cache(16, 8, 10000.0, torch.device("cpu"), torch.float32)
    x = torch.randn(2, 8, 64)
    assert layer(x, cos, sin).shape == (2, 8, 64)


def test_the_mlp_follows_the_swiglu_formula():
    mlp = MLP(hidden_size=8, intermediate_size=16)
    x = torch.randn(1, 3, 8)
    expected = mlp.down_proj(torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    assert torch.allclose(mlp(x), expected, atol=1e-6)


def test_the_attention_hides_the_future():
    torch.manual_seed(0)
    attention = Attention(hidden_size=64, num_heads=4)
    cos, sin = build_rope_cache(16, 8, 10000.0, torch.device("cpu"), torch.float32)
    x = torch.randn(1, 8, 64)
    changed = x.clone()
    changed[:, 5:, :] = torch.randn(1, 3, 64)
    before = attention(x, cos, sin)
    after = attention(changed, cos, sin)
    assert torch.allclose(before[:, :5], after[:, :5], atol=1e-5)
    assert not torch.allclose(before[:, 5:], after[:, 5:], atol=1e-5)
