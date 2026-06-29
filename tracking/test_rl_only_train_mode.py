import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.models import LightFC
from lib.train.actors import LightFCASRActor
from lib.train.loss import lightTrackObjective
from lib.utils.load import load_yaml


def assert_module_eval(module, name):
    if module is not None:
        assert module.training is False, f"{name} should be eval() in rl_only"


def assert_module_train(module, name):
    if module is not None:
        assert module.training is True, f"{name} should be train() in rl_only"


def assert_params_frozen(module, name):
    if module is not None:
        for param_name, param in module.named_parameters():
            assert param.requires_grad is False, f"{name}.{param_name} should be frozen in rl_only"


def assert_params_trainable(module, name):
    if module is not None:
        for param_name, param in module.named_parameters():
            assert param.requires_grad is True, f"{name}.{param_name} should be trainable in rl_only"


def parse_args():
    parser = argparse.ArgumentParser(description="Check RL-only train/eval mode split.")
    parser.add_argument(
        "--cfg",
        default=os.path.join(PROJECT_ROOT, "experiments", "lightfc_asr", "lightfc_asr_tinyvit.yaml"),
        help="Path to an rl_only LightFC ASR config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.cfg)
    cfg.TRAIN.TYPE = "rl_only"
    cfg.MODEL.ASR_ENABLE = True
    cfg.MODEL.BACKBONE.USE_PRETRAINED = False

    net = LightFC(cfg, training=False)
    actor = LightFCASRActor(
        net=net,
        objective=lightTrackObjective(cfg),
        loss_weight={},
        settings=None,
        cfg=cfg,
    )

    actor.train(True)
    actor._set_rl_only_train_mode()

    check_net = actor.net.module if hasattr(actor.net, "module") else actor.net

    for name in ["backbone", "fusion", "head"]:
        module = getattr(check_net, name, None)
        assert_module_eval(module, name)
        assert_params_frozen(module, name)

    for name in ["policy_model", "value_model"]:
        module = getattr(check_net, name, None)
        assert_module_train(module, name)
        assert_params_trainable(module, name)

    print("RL-only train/eval mode check passed.")


if __name__ == "__main__":
    main()
