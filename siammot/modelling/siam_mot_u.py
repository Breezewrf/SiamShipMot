"""
Implements the Generalized R-CNN for SiamMOT
"""
from torch import nn

import siammot.operator_patch.run_operator_patch

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn
from .roi_heads import build_roi_heads
from .backbone.backbone_ext import build_backbone, build_U2NET_BACKBONE
from siammot.blocks.side_blk import Side

class SiamMOT(nn.Module):
    """
    Main class for R-CNN. Currently supports boxes and tracks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and
             computes detections / tracks from it.
    """

    def __init__(self, cfg):
        super(SiamMOT, self).__init__()
        # self.backbone = build_backbone(cfg)
        self.backbone = build_U2NET_BACKBONE()
        self.side = Side(self.backbone.out_channels)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.track_memory = None

    def flush_memory(self, cache=None):
        self.track_memory = cache

    def reset_siammot_status(self):
        self.flush_memory()
        self.roi_heads.reset_roi_status()

    def forward(self, images, targets=None, given_detection=None):
        # target: ground truth
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        features = self.backbone(images.tensors)
        features_sided = self.side(features)
        proposals, proposal_losses = self.rpn(images, features_sided, targets)

        if self.roi_heads:
            x, result, roi_losses = self.roi_heads(features,
                                                   proposals,
                                                   targets,
                                                   self.track_memory,
                                                   given_detection)
            if not self.training:
                # in testing mode, the search region is depended on tracking results in last frame including ids,
                # rather than expand from template like training
                self.flush_memory(cache=x)

        else:
            raise NotImplementedError

        if self.training:
            losses = {}
            losses.update(roi_losses)
            losses.update(proposal_losses)
            return result, losses

        return result


def build_siammot(cfg):
    siammot = SiamMOT(cfg)
    return siammot
