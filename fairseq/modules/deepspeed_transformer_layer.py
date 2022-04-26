from dataclasses import dataclass, field
from typing import Optional
from deepspeed import DeepSpeedTransformerLayer
from deepspeed import DeepSpeedTransformerConfig as DSConfig
from omegaconf import II
from regex import P
from torch import Tensor
import torch
from fairseq.models.transformer.transformer_config import TransformerConfig
from fairseq.utils import safe_getattr, safe_hasattr

@dataclass
class DeepSpeedTransformerConfig(TransformerConfig):
    fp16: bool = II("common.fp16")
    seed: bool = II("common.seed")
    batch_size: bool = II("dataset.batch_size")
    deepspeed_stochastic_mode: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable for high performance, please note that this flag has some level of non-determinism "
                "and can produce different results on different runs."
                "However, we have seen that by enabling it, the pretraining tasks such as BERT are not affected "
                "and can obtain a high accuracy level. On the other hand, "
                "for the downstream tasks, such as fine-tuning, we recommend to turn it off in order to be able "
                "to reproduce the same result through the regular kernel execution."
            )
        }
    )

class DeepSpeedTransformerEncoderLayer(DeepSpeedTransformerLayer):
    def __init__(self, cfg):
        assert cfg.activation_fn == "gelu", "DeepSpeed only supports gelu activation"
        cuda_config = DSConfig(
            hidden_size=cfg.encoder.embed_dim,
            intermediate_size=cfg.encoder.ffn_embed_dim,
            heads=cfg.encoder.attention_heads,
            attn_dropout_ratio=cfg.attention_dropout,
            hidden_dropout_ratio=cfg.dropout,
            num_hidden_layers=cfg.encoder.layers,
            pre_layer_norm=cfg.encoder.normalize_before,
            initializer_range=0.02,
            local_rank=-1,
            layer_norm_eps=1e-5,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            fp16=cfg.fp16,
            normalize_invertible=False,
            attn_dropout_checkpoint=False,
            gelu_checkpoint=safe_getattr(cfg, "checkpoint_activations", False),
            stochastic_mode=cfg.deepspeed_stochastic_mode)
        super().__init__(cuda_config)

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ):
        assert attn_mask is None, "Deepspeed currently doesn't support square attention masks."

        x = x.transpose(0,1)  # T x B x C -> B x T x C

        attention_mask = torch.zeros(x.size(0), 1, 1, x.size(1), dtype=x.dtype, device=x.device)

        if encoder_padding_mask is not None:
            attention_mask = attention_mask.masked_fill_(
                encoder_padding_mask.view(attention_mask.shape), float('-inf'))

        x = super().forward(x, attention_mask=attention_mask)
        return x.transpose(0,1)  # B x T x C -> T x B x C

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Renaming old states doesn't apply to deepspeed layer.
        """
        pass

    def train(self, mode: bool = True):
        self.config.training = mode
        super().train(mode)

    def eval(self):
        self.config.training = False
        super().eval()
