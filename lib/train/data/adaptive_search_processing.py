import random

import torch
import torch.nn.functional as F
from lib.utils import TensorDict
from lib.utils.adaptive_search import ACTIONS
import lib.train.data.processing_utils as prutils
from lib.train.data.processing import BaseProcessing, stack_tensors


class MultiFactorWarmupProcessing(BaseProcessing):
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 factors=ACTIONS, mode="sequence", settings=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.factors = tuple(float(f) for f in factors)
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        if self.transform["joint"] is not None:
            data["template_images"], data["template_anno"], data["template_masks"] = self.transform["joint"](
                image=data["template_images"], bbox=data["template_anno"], mask=data["template_masks"])
            data["search_images"], data["search_anno"], data["search_masks"] = self.transform["joint"](
                image=data["search_images"], bbox=data["search_anno"], mask=data["search_masks"], new_roll=False)

        for source in ["template", "search"]:
            assert self.mode == "sequence" or len(data[source + "_images"]) == 1
            jittered_anno = [self._get_jittered_box(a, source) for a in data[source + "_anno"]]
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            if source == "search":
                factors = [random.choice(self.factors) for _ in data[source + "_images"]]
                crop_sz = torch.ceil(torch.sqrt(w * h) * torch.tensor(factors))
            else:
                factors = [float(self.search_area_factor[source])] * len(data[source + "_images"])
                crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[source])
            if (crop_sz < 1).any():
                data["valid"] = False
                return data

            crops, boxes, att_mask, mask_crops = self._crop_with_factors(
                data[source + "_images"], jittered_anno, data[source + "_anno"],
                factors, self.output_sz[source], masks=data[source + "_masks"])

            data[source + "_images"], data[source + "_anno"], data[source + "_att"], data[source + "_masks"] = \
                self.transform[source](image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            if source == "search":
                data["search_factor"] = torch.tensor(factors, dtype=torch.float32)

            for ele in data[source + "_att"]:
                if (ele == 1).all():
                    data["valid"] = False
                    return data
                feat_size = self.output_sz[source] // 16
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data["valid"] = False
                    return data

        data["valid"] = True
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        return data.apply(stack_tensors) if self.mode == "sequence" else data.apply(
            lambda x: x[0] if isinstance(x, list) else x)

    @staticmethod
    def _crop_with_factors(frames, box_extract, box_gt, factors, output_sz, masks=None):
        if masks is None:
            crops_resize_factors = [prutils.sample_target(f, a, factor, output_sz)
                                    for f, a, factor in zip(frames, box_extract, factors)]
            frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
            masks_crop = None
        else:
            crops_resize_factors = [prutils.sample_target(f, a, factor, output_sz, m)
                                    for f, a, factor, m in zip(frames, box_extract, factors, masks)]
            frames_crop, resize_factors, att_mask, masks_crop = zip(*crops_resize_factors)
        crop_sz = torch.Tensor([output_sz, output_sz])
        box_crop = [prutils.transform_image_to_crop(a_gt, a_ex, rf, crop_sz, normalize=True)
                    for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]
        return frames_crop, box_crop, att_mask, masks_crop


class AdaptiveSearchCandidateProcessing(BaseProcessing):
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 factors=ACTIONS, mode="sequence", settings=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.factors = tuple(float(f) for f in factors)
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        if self.transform["joint"] is not None:
            data["template_images"], data["template_anno"], data["template_masks"] = self.transform["joint"](
                image=data["template_images"], bbox=data["template_anno"], mask=data["template_masks"])
            data["search_images"], data["search_anno"], data["search_masks"] = self.transform["joint"](
                image=data["search_images"], bbox=data["search_anno"], mask=data["search_masks"], new_roll=False)

        template_jittered = [self._get_jittered_box(a, "template") for a in data["template_anno"]]
        crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(
            data["template_images"], template_jittered, data["template_anno"],
            self.search_area_factor["template"], self.output_sz["template"], masks=data["template_masks"])
        data["template_images"], data["template_anno"], data["template_att"], data["template_masks"] = \
            self.transform["template"](image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

        candidate_images = []
        candidate_annos = []
        candidate_atts = []
        candidate_factors = []
        candidate_extract_boxes = []
        search_original_anno = list(data["search_anno"])

        for t, (frame, gt_box) in enumerate(zip(data["search_images"], data["search_anno"])):
            prev_idx = max(t - 1, 0)
            extract_box = self._get_jittered_box(data["search_anno"][prev_idx], "search")
            frame_candidates, anno_candidates, att_candidates = [], [], []
            for action_id, factor in enumerate(self.factors):
                crop, resize_factor, att = prutils.sample_target(frame, extract_box, factor, self.output_sz["search"])
                crop_sz = torch.Tensor([self.output_sz["search"], self.output_sz["search"]])
                anno_crop = prutils.transform_image_to_crop(gt_box, extract_box, resize_factor, crop_sz, normalize=True)
                image_t, anno_t, att_t = self.transform["search"](
                    image=[crop], bbox=[anno_crop], att=[att], joint=True, new_roll=(action_id == 0))
                frame_candidates.append(image_t[0])
                anno_candidates.append(anno_t[0])
                att_candidates.append(att_t[0])
            candidate_images.append(torch.stack(frame_candidates, dim=0))
            candidate_annos.append(torch.stack(anno_candidates, dim=0))
            candidate_atts.append(torch.stack(att_candidates, dim=0))
            candidate_factors.append(torch.tensor(self.factors, dtype=torch.float32))
            candidate_extract_boxes.append(extract_box)

        data["search_images_candidates"] = torch.stack(candidate_images, dim=0)
        data["search_anno_candidates"] = torch.stack(candidate_annos, dim=0)
        data["search_att_candidates"] = torch.stack(candidate_atts, dim=0)
        data["search_factor_candidates"] = torch.stack(candidate_factors, dim=0)
        data["search_extract_anno"] = torch.stack(candidate_extract_boxes, dim=0)
        data["search_original_anno"] = torch.stack(search_original_anno, dim=0)

        data["search_images"] = data["search_images_candidates"]
        data["search_anno"] = data["search_anno_candidates"]
        data["search_masks"] = torch.zeros((len(candidate_images), len(self.factors),
                                            self.output_sz["search"], self.output_sz["search"]))
        data["valid"] = True

        for ele in data["template_att"]:
            if (ele == 1).all():
                data["valid"] = False
                return data

        # =========================
        # SAFE OUTPUT VERSION - fixed
        # =========================

        if data["template_masks"] is None:
            data["template_masks"] = torch.zeros(
                (len(data["template_images"]),
                 self.output_sz["template"],
                 self.output_sz["template"])
            )

        if data["search_masks"] is None:
            data["search_masks"] = torch.zeros(
                (len(data["search_images"]),
                 self.output_sz["search"],
                 self.output_sz["search"])
            )

        def keep_for_training(v):
            if isinstance(v, torch.Tensor):
                return True
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                return True
            if isinstance(v, bool):
                return True
            return False

        clean_data = TensorDict({
            k: v for k, v in data.items()
            if keep_for_training(v)
        })

        if self.mode == "sequence":
            return clean_data.apply(stack_tensors)
        else:
            return clean_data.apply(lambda x: x[0] if isinstance(x, list) else x)