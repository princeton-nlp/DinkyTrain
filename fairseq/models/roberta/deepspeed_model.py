# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.roberta import (
    RobertaModel,
    RobertaEncoder,
    base_architecture,
    roberta_prenorm_architecture,
    roberta_base_architecture,
    roberta_large_architecture
)
from fairseq.models.transformer import TransformerEncoder
from fairseq.models.transformer.transformer_encoder import TransformerEncoderBase
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import safe_getattr, safe_hasattr

logger = logging.getLogger(__name__)


@register_model("deepspeed_roberta")
class DeepSpeedRobertaModel(RobertaModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--deepspeed-stochastic-mode",
            action="store_true",
            help=(
                "Enable for high performance, please note that this flag has some level of non-determinism "
                "and can produce different results on different runs."
                "However, we have seen that by enabling it, the pretraining tasks such as BERT are not affected "
                "and can obtain a high accuracy level. On the other hand, "
                "for the downstream tasks, such as fine-tuning, we recommend to turn it off in order to be able "
                "to reproduce the same result through the regular kernel execution."
            )
        )
        RobertaModel.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        from omegaconf import OmegaConf

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, False)

        # make sure all arguments are present
        base_architecture(args)

        if not safe_hasattr(args, "max_positions"):
            if not safe_hasattr(args, "tokens_per_sample"):
                args.tokens_per_sample = task.max_positions()
            args.max_positions = args.tokens_per_sample

        encoder = DeepSpeedRobertaEncoder(args, task.source_dictionary)

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, True)

        return cls(args, encoder)


class DeepSpeedRobertaEncoder(RobertaEncoder):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder


class TransformerEncoder(TransformerEncoderBase):
    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        from fairseq.modules.deepspeed_transformer_layer import DeepSpeedTransformerConfig
        self.args = args
        super().__init__(
            DeepSpeedTransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )

    def build_encoder_layer(self, cfg):
        from fairseq.modules.deepspeed_transformer_layer import DeepSpeedTransformerEncoderLayer
        return DeepSpeedTransformerEncoderLayer(cfg)


def deepspeed_base_architecture(args):
    base_architecture(args)


@register_model_architecture("deepspeed_roberta", "deepspeed_roberta_prenorm")
def deepspeed_roberta_prenorm_architecture(args):
    roberta_prenorm_architecture(args)


@register_model_architecture("deepspeed_roberta", "deepspeed_roberta_base")
def deepspeed_roberta_base_architecture(args):
    roberta_base_architecture(args)


@register_model_architecture("deepspeed_roberta", "deepspeed_roberta_large")
def deepspeed_roberta_large_architecture(args):
    roberta_large_architecture(args)

