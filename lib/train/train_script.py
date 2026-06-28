import importlib
import os
import random

import numpy as np
import torch

from lib.models import *
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.train.actors import *
from lib.train.data.base_functions import *
from lib.train.trainers import LTRTrainer
from lib.utils.load import load_yaml
from lib.train.loss import lightTrackObjective


def _unwrap_ddp(net):
    return net.module if hasattr(net, "module") else net


def load_warmup_checkpoint_for_asr(net, ckpt_path):
    if ckpt_path is None or ckpt_path == "":
        return
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("net", checkpoint.get("state_dict", checkpoint))
    target_net = _unwrap_ddp(net)
    missing, unexpected = target_net.load_state_dict(state_dict, strict=False)
    allowed_prefixes = ("policy_model", "value_model", "asr_actor_critic")
    allowed_missing = [k for k in missing if k.startswith(allowed_prefixes)]
    real_missing = [k for k in missing if k not in allowed_missing]
    if real_missing:
        raise RuntimeError(f"Unexpected missing keys when loading ASR warmup checkpoint: {real_missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading ASR warmup checkpoint: {unexpected}")
    print("Loaded ASR warmup checkpoint:", ckpt_path)


def run(settings):
    settings.description = 'Training script'

    cfg = load_yaml(settings.cfg_file)
    print('CFG', cfg)
    update_settings(settings, cfg)

    # init seed
    random.seed(cfg.TRAIN.SEED)
    np.random.seed(cfg.TRAIN.SEED)
    torch.manual_seed(cfg.TRAIN.SEED)
    torch.cuda.manual_seed(cfg.TRAIN.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # Create network
    if settings.script_name in ("lightfc", "lightfc_asr_warmup_tinyvit", "lightfc_asr"):
        net = LightFC(cfg, env_num=settings.env_num, training=True)

    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")

    if getattr(cfg.TRAIN, "TYPE", "normal") == "rl_only":
        load_warmup_checkpoint_for_asr(net, getattr(cfg.TRAIN, "PRV_CKPT", None))

    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")

    # Actors
    if settings.script_name in ("lightfc", "lightfc_asr_warmup_tinyvit"):
        objective = lightTrackObjective(cfg)
        loss_weight = {'iou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': cfg.TRAIN.LOC_WEIGHT, }
        actor = lightTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    elif settings.script_name == "lightfc_asr":
        objective = lightTrackObjective(cfg)
        loss_weight = {}
        actor = LightFCASRActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)

    else:
        raise ValueError("illegal script name")

    # SWA
    settings.use_swa = getattr(cfg.TRAIN, 'USE_SWA', False)
    settings.swa_epoch = getattr(cfg.TRAIN, 'SWA_EPOCH', None)

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    loaders = [loader_train] if getattr(cfg.TRAIN, "TYPE", "normal") == "rl_only" else [loader_train, loader_val]
    trainer = LTRTrainer(actor, loaders, optimizer, settings, lr_scheduler, use_amp=use_amp, )

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=(getattr(cfg.TRAIN, "TYPE", "normal") != "rl_only"), fail_safe=True)
