"""Tests for LiveLoRA-Delta generation controller."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
gudhi = pytest.importorskip("gudhi")

from livelora.core.gen_controller import ChunkMetrics, DeltaConfig, GenerationController
from livelora.core.lora_adapter import LiveLoraConfig, LiveLoraModel, get_lora_target_modules
from livelora.topology.ph_tracker import TopologyState


@pytest.fixture
def tiny_model():
    """Create a tiny GPT-2 for fast testing."""
    from transformers import AutoModelForCausalLM, GPT2Config

    config = GPT2Config(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_positions=128,
    )
    return AutoModelForCausalLM.from_config(config)


@pytest.fixture
def lora_model(tiny_model):
    target_modules = get_lora_target_modules(tiny_model)
    config = LiveLoraConfig(rank=4, target_modules=target_modules)
    return LiveLoraModel(tiny_model, config)


class TestDeltaConfig:
    def test_defaults(self):
        cfg = DeltaConfig()
        assert cfg.chunk_size == 32
        assert cfg.epsilon_kl == 1e-3
        assert cfg.tau_rho == 50.0


class TestGenerationController:
    def test_init(self, lora_model):
        controller = GenerationController(lora_model)
        assert controller.model is lora_model

    def test_try_update_returns_metrics(self, lora_model):
        config = DeltaConfig(max_points=16, max_dimension=0)
        controller = GenerationController(lora_model, config)

        # Set up initial state
        input_ids = torch.randint(0, 100, (1, 20))
        lora_model.checkpoint()
        controller.tracker.reset()

        with torch.no_grad():
            _, baseline_points = controller._get_activations(input_ids)
            controller.tracker.set_baseline(baseline_points)

        metrics = controller._try_update(input_ids, None, chunk_idx=0, topo_state=TopologyState.DRIFTING)

        assert isinstance(metrics, ChunkMetrics)
        assert isinstance(metrics.accepted, bool)
        assert isinstance(metrics.rho, float)
        assert isinstance(metrics.kl_divergence, float)

    def test_generate_produces_tokens(self, lora_model):
        config = DeltaConfig(
            chunk_size=8,
            max_new_tokens=16,
            max_points=16,
            max_dimension=0,
            # Relax gate for testing
            epsilon_kl=1.0,
            tau_rho=0.0,
        )
        controller = GenerationController(lora_model, config)

        input_ids = torch.randint(0, 100, (1, 10))
        output_ids, metrics = controller.generate(
            input_ids, pad_token_id=0,
        )

        # Should have generated some tokens
        assert output_ids.shape[1] > input_ids.shape[1]
        # Should have chunk metrics
        assert isinstance(metrics, list)

    def test_rollback_on_reject(self, lora_model):
        """With very tight gate, updates should be rejected and LoRA unchanged."""
        config = DeltaConfig(
            chunk_size=8,
            max_new_tokens=8,
            max_points=16,
            max_dimension=0,
            epsilon_kl=0.0,  # Impossible to satisfy
            tau_rho=1e10,  # Impossible to satisfy
        )
        controller = GenerationController(lora_model, config)

        input_ids = torch.randint(0, 100, (1, 10))
        lora_model.checkpoint()

        output_ids, metrics = controller.generate(
            input_ids, pad_token_id=0,
        )

        # All updates should have been rejected
        for m in metrics:
            assert not m.accepted

        # LoRA should be unchanged from checkpoint
        drift = lora_model.lora_l2_from_checkpoint()
        assert drift.item() < 1e-8
