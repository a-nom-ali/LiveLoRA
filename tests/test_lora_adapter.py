"""Basic tests for LiveLoRA adapter wrapper."""

import pytest
import torch

# These tests require transformers + peft; skip if not available
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from livelora.core.lora_adapter import LiveLoraConfig, LiveLoraModel


@pytest.fixture
def tiny_model():
    """Create a tiny GPT-2 for fast testing."""
    from transformers import AutoModelForCausalLM, GPT2Config

    config = GPT2Config(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_positions=32,
    )
    return AutoModelForCausalLM.from_config(config)


@pytest.fixture
def lora_model(tiny_model):
    config = LiveLoraConfig(rank=4, target_modules=["c_attn"])
    return LiveLoraModel(tiny_model, config)


class TestLiveLoraModel:
    def test_only_lora_trainable(self, lora_model):
        for name, param in lora_model.model.named_parameters():
            if "lora_" in name:
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_lora_parameters_nonempty(self, lora_model):
        params = lora_model.lora_parameters()
        assert len(params) > 0

    def test_checkpoint_restore(self, lora_model):
        lora_model.checkpoint()

        # Modify LoRA weights
        for p in lora_model.lora_parameters():
            p.data.add_(torch.randn_like(p.data))

        # Check they changed
        drift = lora_model.lora_l2_from_checkpoint()
        assert drift.item() > 0

        # Restore
        lora_model.restore()
        drift_after = lora_model.lora_l2_from_checkpoint()
        assert drift_after.item() < 1e-10

    def test_forward(self, lora_model):
        input_ids = torch.randint(0, 100, (1, 10))
        output = lora_model(input_ids=input_ids)
        assert output.logits.shape == (1, 10, 100)

    def test_get_layer_activations(self, lora_model):
        input_ids = torch.randint(0, 100, (1, 10))
        activations = lora_model.get_layer_activations(input_ids, layer_indices=[0, 1])
        assert 0 in activations
        assert 1 in activations
        assert activations[0].shape == (1, 10, 64)
