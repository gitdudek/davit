# ---------------------------------------------------------
# IOU Tracker
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Licensed under The MIT License [see LICENSE for details]
# Written by Erik Bochinski
# ---------------------------------------------------------

import numpy as np
import csv
import cv2

def load_mot(detections):
    """
    Loads detections stored in a mot-challenge like formatted CSV or numpy array (fieldNames = ['frame', 'id', 'x', 'y',
    'w', 'h', 'score']).

    Args:
        detections

    Returns:
        list: list containing the detections for each frame.
    """

    data = []
    if type(detections) is str:
        raw = np.genfromtxt(detections, delimiter=',', dtype=np.float32)
    else:
        # assume it is an array
        assert isinstance(detections, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
        raw = detections.astype(np.float32)

    start_frame = int(np.min(raw[:, 0]))        #line added
    end_frame = int(np.max(raw[:, 0]))

    #for i in range(1, end_frame+1):        
    for i in range(start_frame, end_frame+1):  #line added
        idx = raw[:, 0] == i
        #print('index is {}'.format(i))
        bbox = raw[idx, 2:6]
        bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
        scores = raw[idx, 6]
        dets = []
        for bb, s in zip(bbox, scores):
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s})
        data.append(dets)

    return data


def save_to_csv(out_path, tracks):
    """
    Saves tracks to a CSV file.

    Args:
        out_path (str): path to output csv file.
        tracks (list): list of tracks to store.
    """

    with open(out_path, "w") as ofile:
        field_names = ['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'wx', 'wy', 'wz']

        odict = csv.DictWriter(ofile, field_names)
        id_ = 1
        for track in tracks:
            for i, bbox in enumerate(track['bboxes']):
                row = {'id': id_,
                       'frame': track['start_frame'] + i,
                       'x': bbox[0],
                       'y': bbox[1],
                       'w': bbox[2] - bbox[0],
                       'h': bbox[3] - bbox[1],
                       'score': track['max_score'],
                       'wx': -1,
                       'wy': -1,
                       'wz': -1}

                odict.writerow(row)
            id_ += 1


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

def get_template(img, bbox):
    """
    Cutting template out of image by using the bbox informations
    bbox format: [bb_left,bb_top,bb_width,bb_height]
    """
    bbox = [int(i) for i in bbox]
    template = img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    
    return template

def template_matching(img, tmplt, tmplt_bbox, factor, meth_idx):
    '''
    Template Matching in defined scope of an image.
    Input parameters:
        - img = image to detect template
        - tmplt = template to detect
        - tmplt_box = bbox of last template in format [bb_left,bb_top,bb_w,bb_h] 
        - factor = scope size as a factor of the template size
        - meth_idx = choose between template matching methods (0 to 5)
    '''
    img_h, img_w = img.shape[:2]
    tmplt_h, tmplt_w = tmplt.shape[:2]
    
    # determine scope dimensions
    scope_w = int(tmplt_w + (tmplt_w * factor * 2))
    scope_h = int(tmplt_h + (tmplt_h * factor * 2))
    
    # locating scope in image
    scope_top = int(tmplt_bbox[1] - (0.5 * (scope_h - tmplt_h)))
    scope_left = int(tmplt_bbox[0] - (0.5 * (scope_w - tmplt_w)))
    
    if scope_w > img_w or scope_h > img_h:
        scope_w, scope_h = img_w, img_h
        scope_top, scope_left = 0, 0
    
    # separate scope    
    scope = img[scope_top:scope_top + scope_h, scope_left:scope_left + scope_w]
    
    # template matching
    res = cv2.matchTemplate(scope,tmplt,meth_idx)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(min_val, max_val)
    
    if meth_idx == 4 or meth_idx == 5:
        match_topleft = min_loc
        tm_conf = 1 - min_val
    else:
        match_topleft = max_loc
        tm_conf = max_val

    img_bbox_left = scope_left + match_topleft[0] 
    img_bbox_top = scope_top + match_topleft[1]
    img_bbox_w = tmplt_w
    img_bbox_h = tmplt_h
    
    return [img_bbox_left, img_bbox_top, img_bbox_w, img_bbox_h], tm_conf