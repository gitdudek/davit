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
        #bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
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
                       #'x': int(bbox[0]),
                       'y': bbox[1],
                       #'y': int(bbox[1]),
                       #'w': bbox[2] - bbox[0],
                       'w': bbox[2],
                       #'h': bbox[3] - bbox[1],
                       'h': bbox[3],
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
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,w,h.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,w,h.

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
    #overlap_x1 = min(x1_1, x1_2)
    overlap_x1 = min(x1_1 + x0_1, x1_2 + x0_2)
    #overlap_y1 = min(y1_1, y1_2)
    overlap_y1 = min(y1_1 + y0_1, y1_2 + y0_2)


    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    #size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_1 = x1_1 * y1_1
    #size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_2 = x1_2 * y1_2
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

def get_template(img, bbox):
    """
    Cutting template out of image by using the bbox informations
    bbox format: [bb_left,bb_top,bb_width,bb_height]
    """
    bbox = [int(i) for i in bbox]

    bbox[0] = max(0, bbox[0])
    bbox[1] = max(0, bbox[1])

    if bbox[0]+bbox[2] > img.shape[1]:
        bbox[2] = img.shape[1] - bbox[0]
    if bbox[1]+bbox[3] > img.shape[0]:
        bbox[3] = img.shape[0] - bbox[1]

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
    scope_w = int(tmplt_w + (tmplt_w * factor))
    scope_h = int(tmplt_h + (tmplt_h * factor))
    
    # locating scope in image
    scope_top = max(0, int(tmplt_bbox[1] - (0.5 * (scope_h - tmplt_h))))
    scope_left = max(0, int(tmplt_bbox[0] - (0.5 * (scope_w - tmplt_w))))
    
    # if scope exceeds image dimensions
    if scope_h + scope_top > img_h:
        scope_h = img_h -scope_top
    if scope_w + scope_left > img_w:
        scope_w = img_w - scope_left
  
    # separate scope    
    scope = img[scope_top:scope_top + scope_h, scope_left:scope_left + scope_w]
    
    # template matching
    res = cv2.matchTemplate(scope,tmplt,meth_idx)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
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
    
    return {'bbox':[img_bbox_left, img_bbox_top, img_bbox_w, img_bbox_h], 'score':tm_conf}

def merge(main_tracks, front_tracks, rear_tracks, sigma_iou_merge):
    '''
    Merge tracklets of front-, main- and rear-parts using the iou paradigm

    Input parameters:
    - main_tracks = Main tracks to be merged
    - front_tracks = Front tracks of the belonging main tracks
    - rear_tracks = Rear tracks of the belonging main tracks
    - simga_iou_merge = IOU threshold to consider parts for merging

    Output parameters:
    - tracks merged = Resulting merged tracks
    '''
    
    ii = 0

    while ii < len(main_tracks):
        main_track = main_tracks[ii]
        # seperate kcf track with the actual id
        track = [bbox for bbox in rear_tracks if bbox['id']==main_track['id']] # track muss nicht neu bestimmt werden wenn ii sich nicht geändert hat
        potential_assignments = []
        
        for bbox in track:
            # isolate possible assignments of the same frame
            front_tracks_frame = [f_bbox for f_bbox in front_tracks if f_bbox['frame']==bbox['frame']]
            if front_tracks_frame:
                best_match = max(front_tracks_frame, key=lambda x: iou(bbox['bbox'], x['bbox']))
                # assign best match to actual bbox
                bbox.update({'match_id':best_match['id'],'iou':iou(bbox['bbox'],best_match['bbox'])})
                # add best_match to possible assignments
                if len(potential_assignments) == 0:
                    potential_assignments.append({'id':best_match['id'],'iou_ttl':bbox['iou'],'frames_ttl':1})
                else:
                    matched_id = False
                    for pot_assignment in potential_assignments:
                        if pot_assignment['id'] == bbox['match_id']:
                            pot_assignment['iou_ttl'] += bbox['iou']
                            pot_assignment['frames_ttl'] += 1
                            matched_id = True
                            break
                    if matched_id == False:
                        potential_assignments.append({'id':best_match['id'],'iou_ttl':bbox['iou'],'frames_ttl':1})  
            else:
                break

        if potential_assignments:        
            # find the best assignment
            best_assignment = max(potential_assignments, key=lambda x: x['iou_ttl']/x['frames_ttl'])

            # merging
            if (best_assignment['iou_ttl']/best_assignment['frames_ttl']) >= sigma_iou_merge:

                # id des rear_tracks ist main_track['id']
                for merging_track in main_tracks:
                    if merging_track['id'] == best_assignment['id']:
                        break

                kcf_start_frame = main_track['start_frame'] + len(main_track['bboxes'])
                kcf_last_frame = merging_track['start_frame'] - 1

                # main_track + kcf_track
                for frame in range(kcf_start_frame, kcf_last_frame + 1):
                    kcf_bbox = next(bbox for bbox in track if bbox['frame'] == frame)
                    main_track['bboxes'].append(kcf_bbox['bbox'])
                # extended track + assigned iou track
                main_track['bboxes'] += merging_track['bboxes']
                
                # delete rear_track of IOU1
                jj = 0
                while jj < len(rear_tracks):
                    r_track = rear_tracks[jj]
                    if r_track['id'] == main_track['id']:
                        del rear_tracks[jj]
                    else:
                        jj += 1
                # delete front_track of IOU2
                kk = 0
                while kk < len(front_tracks):
                    f_track = front_tracks[kk]
                    if f_track['id'] == best_assignment['id']:
                        del front_tracks[kk]
                    else:
                        kk += 1
                # reassign rear_track of IOU2 to IOU1 
                for rear_track in rear_tracks:
                    if rear_track['id'] == best_assignment['id']:
                        rear_track['id'] = main_track['id']
                # delete IOU2 in main_tracks
                ll = 0
                while ll < len(main_tracks):
                    m_track = main_tracks[ll]
                    if m_track['id']==best_assignment['id']:
                        del main_tracks[ll]
                    else:
                        ll += 1
                # Am Ende der Function alle Tracks neu nummerieren mit IDs
                # --> Evtl. nicht notwendig, da IDs später in save_to_csv verteilt werden!
            else:
                ii += 1
        else:
            ii += 1
    return main_tracks