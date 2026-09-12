import inspect

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM

from xslm.config import build_config, count_non_embedding_parameters
from xslm.model import build_model


def tiny():
    return build_config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        vocab_size=128,
        max_position_embeddings=32,
    )


def test_the_forward_pass_gives_the_right_shapes():
    config = tiny()
    model = build_model(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    out = model(input_ids=input_ids, labels=input_ids)
    assert out.logits.shape == (2, 16, config.vocab_size)
    assert out.loss.ndim == 0


def test_the_state_dictionary_matches_llama_key_for_key():
    config = tiny()
    mine = {k: tuple(v.shape) for k, v in build_model(config).state_dict().items()}
    theirs = {k: tuple(v.shape) for k, v in LlamaForCausalLM(config).state_dict().items()}
    assert mine == theirs


def test_the_embedding_is_tied_to_the_output():
    model = build_model(tiny())
    assert model.lm_head.weight is model.model.embed_tokens.weight
    # PyTorch writes a tied parameter under every registered name, so the key stays.
    state = model.state_dict()
    assert torch.equal(state["lm_head.weight"], state["model.embed_tokens.weight"])


def test_the_loss_equals_a_hand_written_next_token_cross_entropy():
    config = tiny()
    model = build_model(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
        expected = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, config.vocab_size),
            input_ids[:, 1:].reshape(-1),
        )
    assert torch.allclose(out.loss, expected, atol=1e-5)


def test_the_residual_projections_start_smaller_than_the_input_projections():
    model = build_model(tiny())
    layer = model.model.layers[0]
    assert layer.self_attn.o_proj.weight.std() < layer.self_attn.q_proj.weight.std()
    assert layer.mlp.down_proj.weight.std() < layer.mlp.gate_proj.weight.std()


def test_the_production_model_holds_between_49m_and_51m_non_embedding_parameters():
    config = build_config()
    counted = count_non_embedding_parameters(config)
    model = build_model(build_config(vocab_size=64))
    measured = sum(p.numel() for n, p in model.named_parameters() if "embed_tokens" not in n)
    assert 49_000_000 <= counted <= 51_000_000
    assert measured == counted


def test_fifty_steps_on_twenty_tokens_drive_the_loss_below_zero_point_one():
    torch.manual_seed(0)
    config = tiny()
    model = build_model(config).train()
    input_ids = torch.randint(0, config.vocab_size, (1, 20))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss = None
    for _ in range(50):
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1


def test_the_forward_pass_names_the_token_count_that_the_trainer_passes():
    """Trainer reads the signature to decide whether it must scale the loss.

    A forward that hides `num_items_in_batch` inside `**kwargs` makes Trainer skip
    the division by `gradient_accumulation_steps`, so every gradient grows by that
    factor. See trainer.py, the block that normalizes the loss for reporting.
    """
    parameters = inspect.signature(build_model(tiny()).forward).parameters
    assert "num_items_in_batch" in parameters
    assert parameters["num_items_in_batch"].kind is not inspect.Parameter.VAR_KEYWORD


def test_the_token_count_divides_the_summed_loss():
    """The loss must follow the transformers contract.

    With a token count the loss is the sum over the tokens divided by that count.
    Trainer sums the counts of all the micro batches, so this one rule makes the
    accumulated gradient equal the gradient of the whole batch.
    """
    config = tiny()
    model = build_model(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    count = 2 * 15
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids, num_items_in_batch=count)
        summed = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, config.vocab_size),
            input_ids[:, 1:].reshape(-1),
            reduction="sum",
        )
    assert torch.allclose(out.loss, summed / count, atol=1e-5)


def test_the_accumulated_gradient_equals_the_whole_batch_gradient():
    """Two micro batches must give the gradient of the one batch that holds both.

    This is the property that the missing token count broke on the pod.
    """
    config = tiny()
    torch.manual_seed(0)
    model = build_model(config)
    batch = torch.randint(0, config.vocab_size, (4, 16))
    count = 4 * 15

    model.zero_grad()
    model(input_ids=batch, labels=batch).loss.backward()
    whole = model.model.embed_tokens.weight.grad.clone()

    model.zero_grad()
    for half in batch.split(2):
        model(input_ids=half, labels=half, num_items_in_batch=count).loss.backward()
    accumulated = model.model.embed_tokens.weight.grad.clone()

    assert torch.allclose(whole, accumulated, atol=1e-6)
