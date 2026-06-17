import torch

from lib.models import LightFC
from lib.utils.box_ops import clip_box, box_xywh_to_xyxy, box_iou, box_xyxy_to_xywh
from lib.utils.adaptive_search import action_index_to_factor, build_state, compute_apce, compute_motion
from lib.test.utils.hann import hann2d
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.processing_utils import sample_target
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom
from matplotlib import cm


class lightFC(BaseTracker):
    def __init__(self, params, dataset_name):
        super(lightFC, self).__init__(params)

        network = LightFC(cfg=params.cfg, env_num=None, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)

        for module in network.backbone.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        for module in network.head.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()

        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.prev_pred_box = None
        self.current_search_factor = getattr(self.params, "init_search_factor",
                                             getattr(self.params, "search_factor", 4.0))

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE

        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.frame_id = 0

    def initialize(self, image, info: dict):
        H, W, _ = image.shape

        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)

        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        with torch.no_grad():
            self.z_feat = self.network.forward_backbone(template.tensors)

        self.state = info['init_bbox']
        self.prev_pred_box = None
        self.current_search_factor = getattr(self.params, "init_search_factor",
                                             getattr(self.params, "search_factor", 4.0))
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        factor_used = self.current_search_factor
        prev_state = list(self.state)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, factor_used,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)

        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            out_dict = self.network.forward_tracking(z_feat=self.z_feat, x=x_dict.tensors)

        raw_score_map = out_dict['score_map']
        response_origin = self.output_window * raw_score_map

        # ==================== 画 response map-3D 将2D图像逆时针转了45°====================
        # resp_np = response_origin.squeeze().detach().cpu().numpy()
        #
        # # 对比度增强
        # resp_np = np.log1p(resp_np - resp_np.min() + 1e-8)
        #
        # # 插值加密网格，让表面丝滑
        # scale = 8
        # resp_smooth = zoom(resp_np, scale, order=3)
        #
        # H, W = resp_smooth.shape
        # X, Y = np.meshgrid(np.arange(W), np.arange(H))
        # Z = resp_smooth
        #
        # # 画图
        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(111, projection='3d')
        #
        # surf = ax.plot_surface(X, Y, Z, cmap='turbo',
        #                        rstride=1, cstride=1,
        #                        alpha=0.95,
        #                        linewidth=0,
        #                        antialiased=True,
        #                        shade=True)
        #
        # # 彻底去掉底面和背面
        # ax.xaxis.pane.fill = False
        # ax.yaxis.pane.fill = False
        # ax.zaxis.pane.fill = False
        # ax.xaxis.pane.set_edgecolor('none')
        # ax.yaxis.pane.set_edgecolor('none')
        # ax.zaxis.pane.set_edgecolor('none')
        #
        # ax.set_axis_off()  # 去掉所有坐标轴和刻度
        # ax.view_init(elev=35, azim=-45)
        # ax.set_title(f'3D Response Map (Frame {self.frame_id})', fontsize=13, pad=10)
        #
        # plt.tight_layout()
        # plt.savefig(f'bike1/ours/response_3d_smooth_frame_{self.frame_id:04d}.png', dpi=200, bbox_inches='tight', transparent=True)
        # plt.close()
#----------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------2D-------------------------------------------------------------------------

        # resp_np = response_origin.squeeze().detach().cpu().numpy()  # (feat_sz, feat_sz)
        #
        # # 归一化到 0-255
        # resp_norm = cv.normalize(resp_np, None, 0, 255, cv.NORM_MINMAX)
        # resp_uint8 = resp_norm.astype(np.uint8)
        #
        # # 伪彩色热力图
        # heatmap = cv.applyColorMap(resp_uint8, cv.COLORMAP_JET)
        #
        # # resize 到搜索图大小
        # search_img = x_patch_arr.copy()  # 已经是 RGB，转 BGR
        # search_bgr = cv.cvtColor(search_img, cv.COLOR_BGR2RGB)  # 注意通道顺序
        #
        # heatmap_resized = cv.resize(heatmap, (search_bgr.shape[1], search_bgr.shape[0]))
        #
        # # 叠加
        # overlay = cv.addWeighted(search_bgr, 1, heatmap_resized, 0, 0)
        # cv.imwrite(f'bike1/img/response_overlay_frame_{self.frame_id:04d}.png', overlay)









        pred_box_origin = self.compute_box(response_origin, out_dict,
                                           resize_factor).tolist()  # .unsqueeze(dim=0)  # tolist()

        self.state = clip_box(self.map_box_back(pred_box_origin, resize_factor), H, W, margin=2)
        self._update_next_search_factor(raw_score_map, prev_state, self.state, factor_used)

        return {"target_bbox": self.state}

    def _update_next_search_factor(self, raw_score_map, prev_box_xywh, curr_box_xywh, factor_used):
        if not hasattr(self.network, "policy_model"):
            return
        prev_xyxy = box_xywh_to_xyxy(torch.tensor(prev_box_xywh, dtype=torch.float32, device=raw_score_map.device).view(1, 4))
        curr_xyxy = box_xywh_to_xyxy(torch.tensor(curr_box_xywh, dtype=torch.float32, device=raw_score_map.device).view(1, 4))
        factor_tensor = torch.tensor([factor_used], dtype=torch.float32, device=raw_score_map.device)
        apce = compute_apce(raw_score_map)
        motion = compute_motion(prev_xyxy, curr_xyxy)
        state = build_state(apce, motion, factor_tensor)
        with torch.no_grad():
            next_action = self.network.policy_model(state).argmax(dim=-1)
        self.current_search_factor = action_index_to_factor(next_action).item()

    def compute_box(self, response, out_dict, resize_factor):
        pred_boxes = self.network.head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_boxes = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor)
        return pred_boxes

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return lightFC
