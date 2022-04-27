"""
Author: Tianyu Gao
Email: tianyug@cs.princeton.edu

Convert a fairseq checkpoint to a hf checkpoint.

Usage: [PRELAYERNORM=1] [FROM_DS=1] python convert_fs_ckpt_to_hf_ckpt.py --fr {fairseq checkpoint} --to {huggingface checkpoint path} --hf_model_config {roberta-base/roberta-large} 
"""

import argparse
import pathlib

import fairseq
import torch
from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
from fairseq.modules import TransformerSentenceEncoderLayer
from packaging import version

from transformers import RobertaConfig, AutoTokenizer
import os
if os.getenv("PRELAYERNORM") is not None and os.getenv("PRELAYERNORM") == "1":
    print("Use pre layer norm")
    import sys
    sys.path.append("..")
    from huggingface.modeling_roberta_prelayernorm import RobertaForMaskedLM, RobertaForSequenceClassification
    pre_layer_norm = True
else:
    from transformers import RobertaModel as TransformerRobertaModel
    pre_layer_norm = False
from convert_dsfs_ckpt_to_fs_ckpt import convert_dsfs_ckpt_to_fs_ckpt
from transformers.utils import logging
import shutil
import tempfile

if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_TEXT = "Hello world! cécé herlolip"


def convert_fs_ckpt_to_hf_ckpt(
    fs_ckpt_path: str, 
    hf_ckpt_path: str, 
    classification_head: bool, 
    dict_path: str = "dict.txt",
    hf_model_config: str = "roberta-large"
):
    """
    Copy/paste/tweak roberta's weights to our BERT structure.
    """
    dirname = os.path.dirname(fs_ckpt_path)
    if len(dirname) == 0:
        dirname = "."
    try:
        shutil.copy(dict_path, dirname)
    except:
        pass
    roberta = FairseqRobertaModel.from_pretrained(dirname, checkpoint_file=os.path.basename(fs_ckpt_path))
    roberta.eval()  # disable dropout
    roberta_sent_encoder = roberta.model.encoder.sentence_encoder
    config = RobertaConfig.from_pretrained(hf_model_config)
    if classification_head:
        config.num_labels = roberta.model.classification_heads["mnli"].out_proj.weight.shape[0]
    print("Our RoBERTa config:", config)

    model = RobertaForSequenceClassification(config) if classification_head else RobertaForMaskedLM(config)
    model.eval()

    # Now let's copy all the weights.
    # Embeddings
    model.roberta.embeddings.word_embeddings.weight = roberta_sent_encoder.embed_tokens.weight
    model.roberta.embeddings.position_embeddings.weight = roberta_sent_encoder.embed_positions.weight
    model.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.roberta.embeddings.token_type_embeddings.weight
    )  # just zero them out b/c RoBERTa doesn't use them.
    model.roberta.embeddings.LayerNorm.weight = roberta_sent_encoder.layernorm_embedding.weight
    model.roberta.embeddings.LayerNorm.bias = roberta_sent_encoder.layernorm_embedding.bias

    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        layer = model.roberta.encoder.layer[i]
        roberta_layer: TransformerSentenceEncoderLayer = roberta_sent_encoder.layers[i]

        # self attention
        self_attn = layer.attention.self
        assert (
            roberta_layer.self_attn.k_proj.weight.data.shape
            == roberta_layer.self_attn.q_proj.weight.data.shape
            == roberta_layer.self_attn.v_proj.weight.data.shape
            == torch.Size((config.hidden_size, config.hidden_size))
        )

        self_attn.query.weight.data = roberta_layer.self_attn.q_proj.weight
        self_attn.query.bias.data = roberta_layer.self_attn.q_proj.bias
        self_attn.key.weight.data = roberta_layer.self_attn.k_proj.weight
        self_attn.key.bias.data = roberta_layer.self_attn.k_proj.bias
        self_attn.value.weight.data = roberta_layer.self_attn.v_proj.weight
        self_attn.value.bias.data = roberta_layer.self_attn.v_proj.bias

        # self-attention output
        self_output = layer.attention.output
        assert self_output.dense.weight.shape == roberta_layer.self_attn.out_proj.weight.shape
        self_output.dense.weight = roberta_layer.self_attn.out_proj.weight
        self_output.dense.bias = roberta_layer.self_attn.out_proj.bias
        if pre_layer_norm:
            layer.attention.LayerNorm.weight = roberta_layer.self_attn_layer_norm.weight
            layer.attention.LayerNorm.bias = roberta_layer.self_attn_layer_norm.bias
        else:
            self_output.LayerNorm.weight = roberta_layer.self_attn_layer_norm.weight
            self_output.LayerNorm.bias = roberta_layer.self_attn_layer_norm.bias
 
        # intermediate
        intermediate = layer.intermediate
        assert intermediate.dense.weight.shape == roberta_layer.fc1.weight.shape
        intermediate.dense.weight = roberta_layer.fc1.weight
        intermediate.dense.bias = roberta_layer.fc1.bias

        # output
        bert_output = layer.output
        assert bert_output.dense.weight.shape == roberta_layer.fc2.weight.shape
        bert_output.dense.weight = roberta_layer.fc2.weight
        bert_output.dense.bias = roberta_layer.fc2.bias
        if pre_layer_norm:
            layer.intermediate.LayerNorm.weight = roberta_layer.final_layer_norm.weight
            layer.intermediate.LayerNorm.bias = roberta_layer.final_layer_norm.bias
        else:
            bert_output.LayerNorm.weight = roberta_layer.final_layer_norm.weight
            bert_output.LayerNorm.bias = roberta_layer.final_layer_norm.bias
 
        # end of layer

    # The last layer norm layer for pre-layernorm  
    if pre_layer_norm:
        model.roberta.LayerNorm.weight = roberta_sent_encoder.layer_norm.weight
        model.roberta.LayerNorm.bias = roberta_sent_encoder.layer_norm.bias

    if classification_head:
        model.classifier.dense.weight = roberta.model.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = roberta.model.classification_heads["mnli"].dense.bias
        model.classifier.out_proj.weight = roberta.model.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = roberta.model.classification_heads["mnli"].out_proj.bias
    else:
        # LM Head
        model.lm_head.dense.weight = roberta.model.encoder.lm_head.dense.weight
        model.lm_head.dense.bias = roberta.model.encoder.lm_head.dense.bias
        model.lm_head.layer_norm.weight = roberta.model.encoder.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = roberta.model.encoder.lm_head.layer_norm.bias
        model.lm_head.decoder.weight = roberta.model.encoder.lm_head.weight
        model.lm_head.decoder.bias = roberta.model.encoder.lm_head.bias
        model.lm_head.bias = roberta.model.encoder.lm_head.bias

    # Let's check that we get the same results.
    input_ids: torch.Tensor = roberta.encode(SAMPLE_TEXT).unsqueeze(0)  # batch of size 1
    our_output = model(input_ids, output_hidden_states=True)
    our_output_final = our_output[0].cpu()
    our_output = our_output.hidden_states 
    if classification_head:
        their_output = roberta.model.classification_heads["mnli"](roberta.extract_features(input_ids))
    else:
        their_output = roberta.model(input_ids, return_all_hiddens=True)
        their_output_final = their_output[0].cpu()
        their_output = their_output[1]['inner_states'] 

    # for i in range(len(our_output)):
    #     max_absolute_diff = torch.max(torch.abs(our_output[i].cpu() - their_output[i].transpose(0, 1).cpu())).item()
    #     mean_absolute_diff = torch.mean(torch.abs(our_output[i].cpu() - their_output[i].transpose(0, 1).cpu())).item()
    #     print("Layer %d max diff: %f mean diff: %f" % (i, max_absolute_diff, mean_absolute_diff)) 

    max_absolute_diff = torch.max(torch.abs(our_output_final - their_output_final)).item()
    print(f"max_absolute_diff = {max_absolute_diff}") 
    success = torch.allclose(our_output_final, their_output_final, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")
    pathlib.Path(hf_ckpt_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {hf_ckpt_path}")
    model.save_pretrained(hf_ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_config)
    tokenizer.save_pretrained(hf_ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fr", default=None, type=str, required=True, help="Path to the fairseq checkpoint."
    )
    parser.add_argument(
        "--to", default=None, type=str, required=True, help="Path to the output Huggingface folder."
    )
    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    parser.add_argument(
        "--dict_path", default="dict.txt", type=str, help="Roberta dictionary."
    )
    parser.add_argument(
        "--hf_model_config", default="roberta-large", type=str, help="Huggingface roberta config name."
    )

    args = parser.parse_args()

    if os.getenv("FROM_DS") is not None and os.getenv("FROM_DS") == "1":
        # First convert the deepspeed fairseq checkpoint to fairseq checkpoint
        tmp_ckpt = tempfile.mkstemp()[1]
        convert_dsfs_ckpt_to_fs_ckpt(args.fr, tmp_ckpt)
        convert_fs_ckpt_to_hf_ckpt(
            tmp_ckpt, args.to, args.classification_head, dict_path=args.dict_path, hf_model_config=args.hf_model_config
        )
    else:
        convert_fs_ckpt_to_hf_ckpt(
            args.fr, args.to, args.classification_head, dict_path=args.dict_path, hf_model_config=args.hf_model_config
        )
