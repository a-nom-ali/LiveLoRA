"""LoRA adapter wrapper with fine-grained gradient control for test-time training.

Wraps HuggingFace PEFT to provide:
- Selective gradient enable/disable per layer
- Checkpoint/restore for adapt-and-reset pattern
- Per-layer parameter access for ScaleNet integration
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel


@dataclass
class LiveLoraConfig:
    """Configuration for LiveLoRA adapter."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"

    def to_peft_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias=self.bias,
        )


class LiveLoraModel(nn.Module):
    """Wraps a base model with LoRA for test-time adaptation.

    Key features vs vanilla PEFT:
    - `checkpoint()` / `restore()` for adapt-and-reset
    - `lora_parameters()` yields only LoRA params (for optimizer)
    - `freeze_base()` ensures only LoRA is trainable
    """

    def __init__(self, base_model: PreTrainedModel, config: LiveLoraConfig):
        super().__init__()
        peft_config = config.to_peft_config()
        self.model = get_peft_model(base_model, peft_config)
        self.config = config
        self._checkpoint: dict[str, torch.Tensor] | None = None
        self.freeze_base()

    def freeze_base(self):
        """Ensure only LoRA parameters are trainable."""
        for name, param in self.model.named_parameters():
            param.requires_grad = "lora_" in name

    def lora_parameters(self) -> list[nn.Parameter]:
        """Return only LoRA parameters (for optimizer construction)."""
        return [p for n, p in self.model.named_parameters() if "lora_" in n and p.requires_grad]

    def lora_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Return named LoRA parameters (for per-layer LR scheduling)."""
        return [(n, p) for n, p in self.model.named_parameters() if "lora_" in n]

    def checkpoint(self):
        """Save current LoRA state for later restoration (adapt-and-reset)."""
        self._checkpoint = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if "lora_" in name
        }

    def restore(self):
        """Restore LoRA parameters to last checkpoint."""
        if self._checkpoint is None:
            raise RuntimeError("No checkpoint saved. Call checkpoint() first.")
        for name, param in self.model.named_parameters():
            if "lora_" in name and name in self._checkpoint:
                param.data.copy_(self._checkpoint[name])

    def lora_l2_from_checkpoint(self) -> torch.Tensor:
        """Compute L2 distance of current LoRA params from checkpoint (drift regularizer)."""
        if self._checkpoint is None:
            return torch.tensor(0.0)
        total = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if "lora_" in name and name in self._checkpoint:
                total = total + torch.sum((param - self._checkpoint[name].to(param.device)) ** 2)
        return total

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def get_layer_activations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_indices: list[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Run forward pass and capture hidden states at specified layers.

        Returns:
            Dict mapping layer index -> (batch, seq_len, hidden_dim) activation tensor.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states  # tuple of (batch, seq, dim)

        if layer_indices is None:
            # Default: last layer
            layer_indices = [len(hidden_states) - 1]

        return {i: hidden_states[i] for i in layer_indices if i < len(hidden_states)}
