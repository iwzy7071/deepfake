# --------------------------------------------------------
# BoundingBox Rectangle Utils
# Licensed under The MIT License
# Written by limengyao(mengyao.lmy@alibaba-inc.com)
# --------------------------------------------------------
#!/usr/bin/python

import numpy as np


def cxy_wh_2_rect(pos, sz):
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])  # 0-index


def cxy_wh_2_corner(pos, sz):
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, pos[0]+sz[0]/2, pos[1]+sz[1]/2])  # 0-index


def rect_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2]), np.array([rect[2], rect[3]])  # 0-index


def corner_2_cxy_wh(p1, p2):
    return p1[0]+(p2[0]-p1[0])/2, p1[1]+(p2[1]-p1[1])/2, p2[0]-p1[0], p2[1]-p1[1]  # 0-index


def corner_to_rect(box):
    # box = box.copy()
    box = box[:]
    box_ = np.zeros_like(box, dtype=np.float32)
    box_[:, 0] = box[:, 0]
    box_[:, 1] = box[:, 1]
    box_[:, 2] = (box[:, 2] - box[:, 0])
    box_[:, 3] = (box[:, 3] - box[:, 1])
    box_ = box_.astype(np.int)
    return box_


def get_none_rotation_rect(rects):
    if rects.shape[1] == 4:
        return rects
    converted_rects = []
    for bbox in rects:
        x_coord = bbox[0::2]
        y_coord = bbox[1::2]
        x_coord.sort()
        y_coord.sort()
        w = x_coord[-1] - x_coord[0]
        h = y_coord[-1] - y_coord[0]
        bbox = [x_coord[0], y_coord[0], w, h]
        converted_rects.append(bbox)
    return converted_rects


def get_axis_aligned_bbox(region):
    try:
        region = np.array([region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
                           region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]])
    except:
        region = np.array(region)
    cx = np.mean(region[0::2])
    cy = np.mean(region[1::2])
    x1 = min(region[0::2])
    x2 = max(region[0::2])
    y1 = min(region[1::2])
    y2 = max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1
    return cx, cy, w, h


def get_none_rotation_rect(rects):
    if rects.shape[1] == 4:
        return rects
    converted_rects = []
    for bbox in rects:
        x_coord = bbox[0::2]
        y_coord = bbox[1::2]
        x_coord.sort()
        y_coord.sort()
        w = x_coord[-1] - x_coord[0]
        h = y_coord[-1] - y_coord[0]
        bbox = [x_coord[0], y_coord[0], w, h]
        converted_rects.append(bbox)
    return converted_rects


def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    for i in range(rect1.shape[0]):
        if rect1[i, 0] == 0 and rect1[i, 1] == 0 and rect1[i, 2] == 0 and rect1[i, 3] == 0:
            rect1[i] = rect2[i]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def compute_success_overlap(gt_bb, result_bb):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    n_frame = len(gt_bb)
    success = np.zeros(len(thresholds_overlap))
    iou = overlap_ratio(gt_bb, result_bb)
    for i in range(len(thresholds_overlap)):
        success[i] = sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success


def compute_success_error(gt_center, result_center):
    thresholds_error = np.arange(0, 51, 1)
    n_frame = len(gt_center)
    success = np.zeros(len(thresholds_error))
    dist = np.sqrt(np.sum(np.power(gt_center - result_center, 2), axis=1))
    for i in range(len(thresholds_error)):
        success[i] = sum(dist <= thresholds_error[i]) / float(n_frame)
    return success
