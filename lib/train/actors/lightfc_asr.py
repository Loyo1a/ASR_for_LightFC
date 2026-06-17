import torch
from torch.distributions import Categorical

from . import BaseActor
from lib.utils.adaptive_search import (
    action_index_to_factor,
    build_state,
    compute_action_cost,
    compute_apce,
    compute_motion,
    compute_success_auc,
)
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy


class LightFCASRActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.cfg = cfg

        self.alpha_iou = getattr(cfg.TRAIN, "ALPHA_IOU", getattr(cfg.TRAIN, "IOU_REWARD_WEIGHT", 1.0))
        self.beta_auc = getattr(cfg.TRAIN, "BETA_AUC", getattr(cfg.TRAIN, "AUC_REWARD_WEIGHT", 1.0))
        self.eta_cost = getattr(cfg.TRAIN, "ETA_COST", getattr(cfg.TRAIN, "SEARCH_COST_WEIGHT", 0.10))
        self.value_weight = getattr(cfg.TRAIN, "VALUE_WEIGHT", 0.50)
        self.adv_norm = getattr(cfg.TRAIN, "ADV_NORM", True)

    @property
    def net_module(self):
        return self.net.module if hasattr(self.net, "module") else self.net

    @staticmethod
    def _select_candidate(candidates, action_idx):
        batch_idx = torch.arange(candidates.shape[0], device=candidates.device)
        return candidates[batch_idx, action_idx.long()]

    @staticmethod
    def _pred_to_xyxy(pred_boxes):
        if pred_boxes.ndim == 3:
            pred_boxes = pred_boxes[:, 0]
        if pred_boxes.ndim == 1:
            pred_boxes = pred_boxes.view(1, 4)
        return box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)

    def __call__(self, data):
        template = data["template_images"]
        if template.ndim == 5:
            template = template[:, 0]

        search_candidates = data.get("search_images_candidates", data.get("search_images"))
        anno_candidates = data.get("search_anno_candidates", None)
        gt_anno = data.get("search_original_anno", data["search_anno"])

        if search_candidates.ndim != 6:
            raise ValueError(
                "LightFCASRActor expects search candidates with shape [B, T, 4, C, H, W], "
                f"got {tuple(search_candidates.shape)}"
            )

        batch_size, num_frames, num_actions = search_candidates.shape[:3]
        if num_actions != 4:
            raise ValueError(f"Expected four candidate search factors, got {num_actions}")

        log_probs = []
        values = []
        ious = []
        costs = []
        factors_used = []

        prev_pred_box_xyxy = None
        curr_action_idx = torch.ones((batch_size,), dtype=torch.long, device=search_candidates.device)
        curr_factor = action_index_to_factor(curr_action_idx)

        for t in range(num_frames):
            search_t = self._select_candidate(search_candidates[:, t], curr_action_idx)
            out_dict = self.net(z=template, x=search_t)

            pred_xyxy = self._pred_to_xyxy(out_dict["pred_boxes"])
            if anno_candidates is not None:
                gt_box_t = self._select_candidate(anno_candidates[:, t], curr_action_idx)
            else:
                gt_box_t = gt_anno[:, t] if gt_anno.ndim == 3 else gt_anno
            gt_xyxy = box_xywh_to_xyxy(gt_box_t).view(-1, 4).clamp(0.0, 1.0)

            _, iou_t = self.objective.iou(pred_xyxy, gt_xyxy)
            iou_t = iou_t.detach()
            ious.append(iou_t)

            if t > 0:
                factors_used.append(curr_factor)
                costs.append(compute_action_cost(curr_factor))

            apce = compute_apce(out_dict["score_map"])
            if prev_pred_box_xyxy is None:
                motion = torch.zeros((batch_size,), device=search_candidates.device)
            else:
                motion = compute_motion(prev_pred_box_xyxy, pred_xyxy.detach())
            state = build_state(apce.detach(), motion.detach(), curr_factor.detach())

            if t < num_frames - 1:
                logits = self.net_module.policy_model(state)
                value = self.net_module.value_model(state).squeeze(-1)
                dist = Categorical(logits=logits)
                next_action_idx = dist.sample()

                log_probs.append(dist.log_prob(next_action_idx))
                values.append(value)
                curr_action_idx = next_action_idx
                curr_factor = action_index_to_factor(curr_action_idx)

            prev_pred_box_xyxy = pred_xyxy.detach()

        iou_mat = torch.stack(ious, dim=1)
        auc_clip = compute_success_auc(iou_mat).detach()

        rewards = []
        for k in range(num_frames - 1):
            reward = self.alpha_iou * iou_mat[:, k + 1] + self.beta_auc * auc_clip - self.eta_cost * costs[k].detach()
            rewards.append(reward)

        rewards = torch.stack(rewards, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        values = torch.stack(values, dim=1)

        advantage = rewards - values
        if self.adv_norm:
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)

        policy_loss = -(advantage.detach() * log_probs).mean()
        value_loss = (values - rewards.detach()).square().mean()
        total_loss = policy_loss + self.value_weight * value_loss

        factor_tensor = torch.stack(factors_used, dim=1) if factors_used else curr_factor[:, None]
        cost_tensor = torch.stack(costs, dim=1) if costs else torch.zeros_like(factor_tensor)
        status = {
            "Loss/total": total_loss.item(),
            "Loss/policy": policy_loss.item(),
            "Loss/value": value_loss.item(),
            "Reward/mean": rewards.mean().item(),
            "Reward/cost": cost_tensor.mean().item(),
            "IoU/mean": iou_mat.mean().item(),
            "AUC/clip": auc_clip.mean().item(),
            "Policy/MeanFactor": factor_tensor.float().mean().item(),
        }
        return total_loss, status
