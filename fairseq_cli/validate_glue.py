#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from argparse import Namespace
from itertools import chain
import json
import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import metrics, progress_bar
from fairseq import tasks
from omegaconf import DictConfig
from fairseq.logging.meters import safe_round
from fairseq.file_io import PathManager
from filelock import FileLock
from pathlib import Path
import wandb


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


def pearson_and_spearman(preds, labels):
    from scipy.stats import pearsonr, spearmanr
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return pearson_corr, spearman_corr

def get_precision(truepositives, falsepositives, round=3):
    try:
        return safe_round(truepositives / (truepositives + falsepositives), round)
    except (OverflowError, ZeroDivisionError):
        return float("inf")


def get_recall(truepositives, falsenegatives, round=3):
    try:
        return safe_round(truepositives / (truepositives + falsenegatives), round)
    except (OverflowError, ZeroDivisionError):
        return float("inf")


def get_f1(truepositives, falsepositives, falsenegatives, round=3):
    try:
        precision = truepositives / (truepositives + falsepositives)
        recall = truepositives / (truepositives + falsenegatives)
        f1 = 2*(precision*recall)/(precision + recall)
        return safe_round(f1, round)
    except (OverflowError, ZeroDivisionError):
        return float("inf")


def get_matthews_correlation(truepositives, falsepositives, truenegatives, falsenegatives, round=3):
    try:
        numerator = truepositives * truenegatives - falsepositives * falsenegatives
        denominator_squared = (
            (truepositives + falsepositives) *
            (truepositives + falsenegatives) *
            (truenegatives + falsepositives) *
            (truenegatives + falsenegatives))
        return safe_round(numerator / (denominator_squared ** 0.5), round)
    except (OverflowError, ZeroDivisionError):
        return float("inf")


def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))

    if not PathManager.exists(cfg.common_eval.path):
        raise IOError("Model file not found: {}".format(cfg.common_eval.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(cfg.common_eval.path, overrides)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
        state=state,
        # strict=False
    )
    model = models[0]

    optimizer_history = state["optimizer_history"]
    state = None

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Finetuning info
    info = {}
    try:
        # Pretrained model info
        pretraining_step, pretraining_time = (
            max((entry["num_updates"], float(entry.get("cumulative_training_time", 0)))
                for entry in optimizer_history
                if entry["criterion_name"] == "MaskedLmLoss")
        )
        info["finetune_from_model"] = saved_cfg.checkpoint.finetune_from_model
        info["finetune_from_model_name"] = Path(saved_cfg.checkpoint.finetune_from_model).parent.name
        info["finetune_from_model_checkpoint"] = Path(saved_cfg.checkpoint.finetune_from_model).name
    except:
        pretraining_time = 0
        pretraining_step = 0

    info["seed"] = saved_cfg.common.seed
    info["lr"] = saved_cfg.optimization.lr[0]
    info["effective_batch_size"] = (
        saved_cfg.optimization.update_freq[0] * saved_cfg.dataset.batch_size *
        saved_cfg.distributed_training.distributed_world_size
    )
    info["max_update"] = saved_cfg.optimization.max_update
    info["max_epoch"] = saved_cfg.optimization.max_epoch
    info["finetune_steps"] = optimizer_history[-1]["num_updates"]
    info["save_dir"] = saved_cfg.checkpoint.save_dir
    info["save_dir_name"] = Path(saved_cfg.checkpoint.save_dir).name
    info["save_dir_parent"] = Path(saved_cfg.checkpoint.save_dir).parent.name
    info["data_dir"] = saved_cfg.task.data

    info["results_path"] = cfg.common_eval.results_path
    info["results_file"] = (
        Path(cfg.common_eval.results_path).stem
        if cfg.common_eval.results_path else ""
    )

    # Print args
    logger.info(saved_cfg)

    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()

    results = {}
    for subset in cfg.dataset.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1, task_cfg=saved_cfg.task)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        )

        log_outputs = []
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if not sample:
                continue
            model.eval()
            with torch.no_grad():
                logits, _ = model(
                    **sample["net_input"],
                    features_only=True,
                    classification_head_name=criterion.classification_head_name,
                )
                targets = model.get_targets(sample, [logits]).view(-1)
                log_output = {"logits": logits.cpu(),
                              "targets": targets.cpu()}
            log_outputs.append(log_output)

        if data_parallel_world_size > 1:
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size=cfg.common.all_gather_list_size,
                group=distributed_utils.get_data_parallel_group(),
            )
            log_outputs = list(chain.from_iterable(log_outputs))

        logits = torch.cat([output["logits"] for output in log_outputs], dim=0)
        targets = torch.cat([output["targets"] for output in log_outputs], dim=0)
        num_samples = targets.numel()

        log_output = {
            "accuracy": safe_round((logits.argmax(-1) == targets).sum().item() / num_samples, 3),
            "num_samples": num_samples
        }

        if task.cfg.regression_target:
            pearson, spearman = pearson_and_spearman(logits.squeeze(1).numpy(), targets.numpy())
            log_output["pearson"] = pearson
            log_output["spearmanr"] = spearman
        else:
            if task.cfg.num_classes == 2:
                preds = logits.argmax(-1)
                truepositives = (preds[targets==1]).sum().item()
                falsepositives = (preds[targets==0]).sum().item()
                truenegatives = (1-preds[targets==0]).sum().item()
                falsenegatives = (1-preds[targets==1]).sum().item()

                # log_output["truepositives"] = truepositives
                # log_output["falsepositives"] = falsepositives
                # log_output["truenegatives"] = truenegatives
                # log_output["falsenegatives"] = falsenegatives

                log_output["precision"] = get_precision(truepositives, falsepositives)
                log_output["recall"] = get_recall(truepositives, falsenegatives)
                log_output["f1"] = get_f1(truepositives, falsepositives, falsenegatives)
                log_output["matthews_correlation"] = get_matthews_correlation(
                    truepositives, falsepositives, truenegatives, falsenegatives)

        progress.print(log_output, tag=subset, step=i)

        results[subset] = log_output

    if cfg.common.wandb_project:
        pretraining_name = Path(saved_cfg.checkpoint.finetune_from_model).parent.name
        finetuning_name = Path(saved_cfg.checkpoint.save_dir).name
        wandb.init(project=cfg.common.wandb_project,
                   name=(pretraining_name + '-' + finetuning_name),
                   group=pretraining_name, resume="allow", config=info)

        wandb_results = {subset: res for subset, res in results.items()}
        wandb_results["time"] = pretraining_time
        wandb.log(wandb_results, step=pretraining_step)


    if cfg.common_eval.results_path:
        # Add info for json output
        json_results = []
        for subset, res in results.items():
            res["subset"] = subset
            res["pretraining_step"] = pretraining_step
            res["pretraining_time"] = pretraining_time
            res.update(info)
            json_results.append(res)
        lock = FileLock(cfg.common_eval.results_path + ".lock", timeout=30)
        with lock:
            if os.path.exists(cfg.common_eval.results_path):
                with open(cfg.common_eval.results_path, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            data += json_results
            with open(cfg.common_eval.results_path, 'w') as f:
                print(data)
                json.dump(data, f, indent=4)


def cli_main():
    parser = options.get_validation_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(
        override_parser, suppress_defaults=True
    )

    distributed_utils.call_main(
        convert_namespace_to_omegaconf(args), main, override_args=override_args
    )


if __name__ == "__main__":
    cli_main()
