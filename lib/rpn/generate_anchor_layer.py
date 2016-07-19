# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import scipy.io as sio
import caffe, json
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform

DEBUG = False

class GenerateAnchorLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = json.loads(self.param_str_)
        self._scales = layer_params['scales']
        self._base_size = layer_params['base_size']
        self._anchors = generate_anchors(base_size=self._base_size,scales=np.array(self._scales))
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)

        height, width = bottom[0].data.shape[-2:]
        if DEBUG:
            print 'GenerateAnchorLayer: height', height, 'width', width

        A = self._num_anchors
        # bbox_anchors
        top[0].reshape(height * width * A, 4)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].data.shape[-2:]

        rpn_cls_score = bottom[0].data
        #rpn_bbox_pred = bottom[2].data
        # GT boxes (x1, y1, x2, y2, label)
        #gt_boxes = bottom[1].data
        # im_info
        im_info = bottom[1].data[0, :]

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        #print 'shift_x1:',shift_x.shape,shift_x
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        #print 'shift_x2:',shift_x.shape,shift_x
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]

        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))


        # map up to original set of anchors
        bbox_anchors = all_anchors
        
        top[0].reshape(*bbox_anchors.shape)
        top[0].data[...] = bbox_anchors

        #reshape score
        #rpn_cls_score = rpn_cls_score.reshape(2,A,height,width)
        #rpn_cls_score_reshape = rpn_cls_score.transpose(0,2,3,1)
        #rpn_cls_score_reshape = rpn_cls_score_reshape.reshape(1, 2, height * width * A, 1)    
        #top[1].reshape(*rpn_cls_score_reshape.shape)
        #top[1].data[...] = rpn_cls_score_reshape

        #reshape pred
        #rpn_bbox_pred = rpn_bbox_pred.transpose(0,2,3,1)
        #rpn_bbox_pred_reshape = rpn_bbox_pred.reshape(1, 4, height * width * A, 1)
        #top[2].reshape(*rpn_bbox_pred_reshape.shape)
        #top[2].data[...] = rpn_bbox_pred_reshape


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

