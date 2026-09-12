import torch

from xslm.config import build_config
from xslm.model import RMSNorm, XSLMForCausalLM, apply_rope, build_rope_cache

HEAD_DIM, MAX_SEQ = 64, 128


def rope_tables():
    return build_rope_cache(HEAD_DIM, MAX_SEQ, 10000.0, torch.device("cpu"), torch.float32)


def rotate(vector, position):
    """Rotate one vector to one position. apply_rope rotates a pair, so pass it twice."""
    cos, sin = rope_tables()
    rotated, _ = apply_rope(
        vector, vector, cos[position : position + 1], sin[position : position + 1]
    )
    return rotated


def test_rmsnorm_gives_unit_root_mean_square():
    norm = RMSNorm(16, eps=1e-5)
    x = torch.randn(2, 4, 16) * 7.0
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_holds_one_parameter_named_weight():
    names = [name for name, _ in RMSNorm(16, eps=1e-5).named_parameters()]
    assert names == ["weight"]


def test_rope_keeps_the_dot_product_when_both_positions_shift():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)
    near = (rotate(q, 3) * rotate(k, 8)).sum()
    far = (rotate(q, 13) * rotate(k, 18)).sum()
    assert torch.allclose(near, far, atol=1e-4)


def test_rope_changes_the_dot_product_when_the_distance_changes():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)
    close = (rotate(q, 0) * rotate(k, 1)).sum()
    distant = (rotate(q, 0) * rotate(k, 9)).sum()
    assert not torch.allclose(close, distant, atol=1e-3)


def test_the_rope_table_survives_a_save_and_a_load(tmp_path):
    """from_pretrained builds the model on the meta device, then fills it.

    A buffer that the state dictionary does not hold keeps the empty memory, so a
    table that lives only in a non persistent buffer arrives full of noise and every
    position encoding dies. The table must therefore never depend on that memory.
    """
    config = build_config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        vocab_size=128,
        max_position_embeddings=32,
    )
    torch.manual_seed(0)
    model = XSLMForCausalLM(config).eval()
    model.save_pretrained(tmp_path)
    loaded = XSLMForCausalLM.from_pretrained(tmp_path).eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        before = model(input_ids=input_ids, labels=input_ids).loss
        after = loaded(input_ids=input_ids, labels=input_ids).loss
    assert torch.allclose(before, after, atol=1e-5)


def test_the_first_position_leaves_the_vector_unchanged(tmp_path):
    """Position zero has an angle of zero, so its cosine is one and its sine is zero."""
    config = build_config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        vocab_size=128,
        max_position_embeddings=32,
    )
    model = XSLMForCausalLM(config)
    model.save_pretrained(tmp_path)
    loaded = XSLMForCausalLM.from_pretrained(tmp_path)
    cos, sin = loaded.model.rope_tables(torch.device("cpu"), torch.float32)
    assert torch.allclose(cos[0], torch.ones_like(cos[0]))
    assert torch.allclose(sin[0], torch.zeros_like(sin[0]))
