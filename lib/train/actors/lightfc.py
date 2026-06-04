import torch
import torch.nn as nn
from . import BaseActor
from ..loss.cos_sim_loss import cosine_similarity_loss
from ...utils.box_ops import box_xywh_to_xyxy, box_cxcywh_to_xyxy
from ...utils.heapmap_utils import generate_heatmap
import torchvision.transforms as transforms


class lightTrackActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

        # triple loss
        # self.avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        # self.triple = nn.TripletMarginLoss(margin=1, p=2, reduction='mean')

        # self.transform = transforms.RandomErasing(p=0.05, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=0, inplace=False)

    def __call__(self, data):

        out_dict = self.forward_pass(data)

        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][:, i, :].view(-1, *data['template_images'].shape[2:])
            template_list.append(template_img_i)

        #search_img = data['search_images'][:, 0, :].view(-1, *data['search_images'].shape[2:])
        # search_img = self.transform(search_img)

        if len(template_list) == 1:
            template_list = template_list[0]  # [bs, C, H, W]

        # ── 序列前向：逐帧处理全部8帧 ────────────────────────────
        num_search = data['search_images'].shape[1]  # 8
        all_pred = []  # 存储每一帧的预测结果

        for t in range(num_search):
            # [bs, C, H, W]  每次取第t帧
            search_img_t = data['search_images'][:, t, :].view(
                -1, *data['search_images'].shape[2:])

            # 模型推理
            out_dict_t = self.net(z=template_list, x=search_img_t)
            all_pred.append(out_dict_t)

        # all_pred: list of 8 dicts, 每个dict包含 pred_boxes/score_map
        return all_pred

        # out_dict = self.net(z=template_list, x=search_img)
        # return out_dict

    def compute_losses(self, pred_list, gt_dict, return_status=True):
        """
        对序列中每一帧计算损失并聚合
        pred_list: list of 8 pred_dicts
        gt_dict['search_anno']: [bs, 8, 4]
        """

        #bs, n, _ = gt_dict['search_anno'].shape
        #gt_bbox = gt_dict['search_anno'].view(bs, 4) 序列化训练注释
        search_anno = gt_dict['search_anno']  # [bs, 8, 4]
        bs = search_anno.shape[0]
        num_search = len(pred_list)  # 8

        total_loss = 0.0
        total_iou_loss = 0.0
        total_l1_loss = 0.0
        total_loc_loss = 0.0
        total_iou = 0.0

        for t in range(num_search):
            pred_dict = pred_list[t]

            # ── 当前帧GT ──────────────────────────────────────────
            if search_anno.dim() == 3:
                gt_bbox_t = search_anno[:, t, :].contiguous().view(bs, 4)  # [bs,4]
            else:
                gt_bbox_t = search_anno.view(bs, 4)

            # ── Heatmap ───────────────────────────────────────────
            gt_gaussian_maps = generate_heatmap(
                gt_bbox_t.unsqueeze(0),  # [1, bs, 4]
                self.cfg.DATA.SEARCH.SIZE,
                self.cfg.MODEL.BACKBONE.STRIDE
            )
            gt_gaussian_maps_flatten = gt_gaussian_maps[-1].unsqueeze(1)  # [bs,1,H,W]

            # ── 预测值检查 ─────────────────────────────────────────
            pred_boxes = pred_dict['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("NaN detected in pred_boxes at frame %d" % t)

            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # [bs,4]
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox_t).view(-1, 4).clamp(0.0, 1.0)  # [bs,4]

            # ── 逐帧损失 ──────────────────────────────────────────
            try:
                iou_loss_t, iou_t = self.objective.iou(pred_boxes_vec, gt_boxes_vec)
            except:
                iou_loss_t = torch.tensor(0.0).cuda()
                iou_t = torch.tensor(0.0).cuda()

            l1_loss_t = self.objective.l1(pred_boxes_vec, gt_boxes_vec)

            if 'score_map' in pred_dict:
                loc_loss_t = self.objective.focal_loss(
                    pred_dict['score_map'], gt_gaussian_maps_flatten)
            else:
                loc_loss_t = torch.tensor(0.0, device=l1_loss_t.device)

            loss_t = (self.loss_weight['iou'] * iou_loss_t
                      + self.loss_weight['l1'] * l1_loss_t
                      + self.loss_weight['focal'] * loc_loss_t)

            # ── 累积 ──────────────────────────────────────────────
            total_loss += loss_t
            total_iou_loss += iou_loss_t
            total_l1_loss += l1_loss_t
            total_loc_loss += loc_loss_t
            total_iou += iou_t.detach().mean()

            # ── 对8帧取平均 ───────────────────────────────────────────
        total_loss /= num_search
        total_iou_loss /= num_search
        total_l1_loss /= num_search
        total_loc_loss /= num_search
        total_iou /= num_search

        if return_status:
            status = {
                "Loss/total": total_loss.item(),
                "Loss/giou": total_iou_loss.item(),
                "Loss/l1": total_l1_loss.item(),
                "Loss/location": total_loc_loss.item(),
                "mean_IoU": total_iou.item(),
            }
            return total_loss, status
        else:
            return total_loss
