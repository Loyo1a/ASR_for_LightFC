import torch


ACTIONS = (2.0, 4.0, 6.0, 8.0)


def compute_apce(score_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = score_map.float()
    if x.ndim in (3, 4):
        x = x.flatten(1)
    elif x.ndim != 2:
        raise ValueError(f"Unexpected score_map shape: {tuple(score_map.shape)}")

    x_max = x.max(dim=1).values
    x_min = x.min(dim=1).values
    numerator = (x_max - x_min).square()
    denominator = (x - x_min[:, None]).square().mean(dim=1)
    return numerator / (denominator + eps)


def action_index_to_factor(action_idx: torch.Tensor) -> torch.Tensor:
    actions = torch.tensor(ACTIONS, device=action_idx.device, dtype=torch.float32)
    return actions[action_idx.long()]


def factor_to_action_index(factor: torch.Tensor) -> torch.Tensor:
    actions = torch.tensor(ACTIONS, device=factor.device, dtype=torch.float32)
    diff = (factor.float()[..., None] - actions).abs()
    return diff.argmin(dim=-1)


def normalize_factor(factor: torch.Tensor) -> torch.Tensor:
    return (factor.float() - 2.0) / 6.0


def compute_action_cost(factor: torch.Tensor) -> torch.Tensor:
    return normalize_factor(factor).clamp(0.0, 1.0)


def box_center_xyxy(box: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        (box[..., 0] + box[..., 2]) * 0.5,
        (box[..., 1] + box[..., 3]) * 0.5,
    ], dim=-1)


def box_wh_xyxy(box: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        (box[..., 2] - box[..., 0]).clamp(min=1e-6),
        (box[..., 3] - box[..., 1]).clamp(min=1e-6),
    ], dim=-1)


def compute_motion(prev_box_xyxy: torch.Tensor, curr_box_xyxy: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prev_center = box_center_xyxy(prev_box_xyxy)
    curr_center = box_center_xyxy(curr_box_xyxy)
    prev_wh = box_wh_xyxy(prev_box_xyxy)
    denom = torch.sqrt(prev_wh[..., 0] * prev_wh[..., 1]).clamp(min=eps)
    return torch.linalg.norm(curr_center - prev_center, dim=-1) / denom


def compute_success_auc(seq_ious, num_thresholds: int = 21) -> torch.Tensor:
    if isinstance(seq_ious, (list, tuple)):
        iou_mat = torch.stack(seq_ious, dim=1)
    else:
        iou_mat = seq_ious
    if iou_mat.ndim != 2:
        raise ValueError(f"Expected IoU tensor [B, T], got {tuple(iou_mat.shape)}")

    thresholds = torch.linspace(0.0, 1.0, steps=num_thresholds, device=iou_mat.device)
    success = (iou_mat[:, None, :] > thresholds[None, :, None]).float()
    success_rate = success.mean(dim=2)
    return torch.trapz(success_rate, thresholds, dim=1)


def build_state(apce: torch.Tensor, motion: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    q = torch.log1p(apce.float())
    m = motion.float()
    a = normalize_factor(factor.float())
    return torch.stack([q, m, a], dim=-1)
