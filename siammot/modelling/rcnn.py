"""
Implements the Generalized R-CNN for SiamMOT
"""
from torch import nn

import siammot.operator_patch.run_operator_patch

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn
from siammot.blocks.fp2n import fp2n, fp2n_add
from .roi_heads import build_roi_heads
from .backbone.backbone_ext import build_backbone


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

        self.backbone = build_backbone(cfg)
        # self.f2pnet = fp2n_add()
        # f2pnet module
        # self.f2pnet = fp2n()
        # self.f2pnet.init_para()
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.track_memory = None

    def flush_memory(self, cache=None):
        self.track_memory = cache

    def reset_siammot_status(self):
        self.flush_memory()
        self.roi_heads.reset_roi_status()

    def forward(self, images, targets=None, given_detection=None):

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        images = to_image_list(images)
        fp2n_use = False
        if fp2n_use:
            features = self.f2pnet(self.backbone.body(images.tensors))
            proposals, proposal_losses = self.rpn(images, features, targets)
        # features = self.backbone(images.tensors)
        # proposals, proposal_losses = self.rpn(images, features, targets)
        else:   
            features = self.backbone(images.tensors)
            proposals, proposal_losses = self.rpn(images, features, targets)
        # fp2net module
        # features = self.f2pnet(self.backbone.body(images.tensors))
        # det_features = self.backbone(images.tensors)
        # proposals, proposal_losses = self.rpn(images, features, targets)
        
        # global cnt
        # cnt += 1
        # if cnt>=60:
        #     pdb.set_trace()
        if self.roi_heads:
            x, result, roi_losses = self.roi_heads(features,
                                                   proposals,
                                                   targets,
                                                   self.track_memory,
                                                   given_detection)
            if not self.training:
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
