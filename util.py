# ---------------------------------------------------------
# DAVIT Tracker
# Written by Christoph Dudek
# ---------------------------------------------------------

import numpy as np
import csv
import cv2
import math
import os
from itertools import combinations
#from kcf_tracker import read_img
import cProfile, pstats, io


def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


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

def detections_preprocessing(detections,iou_filter):
    '''
    Preprocessing detections. Filters detections for every frame whose IOU overshoots the iou_filter.
    When exceeding the iou_filter the detection with the higher score will be kept.

    Input parameters:
    - detections: Detections to be preprocessed
    - iou_filter: Every detection with iou >= iou_filter to another detection will be filtered

    Output parameters:
    - detections: Filtered detections
    '''

    for dets in detections:
        del_idxs = []
        combs = list(combinations(range(len(dets)),2))
        for comb in combs:
            iou_val = iou(dets[comb[0]]['bbox'],dets[comb[1]]['bbox'])
            if iou_val >= iou_filter:
                if dets[comb[0]]['score'] <= dets[comb[1]]['score']:
                    if not comb[0] in del_idxs:
                        del_idxs.append(comb[0])
                else:
                    if not comb[1] in del_idxs:
                        del_idxs.append(comb[1])
        for del_idx in sorted(del_idxs, reverse=True):
                del dets[del_idx]
    return detections

def get_center(bbox):
    '''
    Get center point of Boundingbox
    '''
    x_center = bbox[0] + (bbox[2]/2)
    y_center = bbox[1] + (bbox[3]/2)

    return (x_center,y_center)

def get_delta(pt1, pt2):
    '''
    Calculating the delta vector between two points (x1,y1), (x2,y2)
    '''
    return np.array((pt2[0] - pt1[0], pt2[1] - pt1[1]))


def get_angle(pt1, pt2):
    '''
    Calculating the angle between two 2d points (x1,y1), (x2,y2)
    Min angle = 0
    Max angle = 180
    '''
    delta_x = pt2[0] - pt1[0]
    delta_y = pt2[1] - pt1[1]

    angle = math.atan2(delta_x, delta_y)/math.pi*180
    #angle = math.degrees(math.atan2(delta_x, delta_y))
    if angle < 0:
        angle = 360 + angle

    return angle

'''def compare_points(pt1, pt2):
    a = np.array(pt1)
    b = np.array([0,0])
    c = np.array(pt2)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)'''

def get_deviation(angle1, angle2):
    '''
    Returns deviation
    '''
    return abs(angle1-angle2)

def scale(value, min_value, max_value, min_scale, max_scale):
    '''
    Scaling a value to given range
    '''
    range_value = (max_value - min_value)  
    range_scale = (max_scale - min_scale)
    return (((value - min_value) * range_scale) / range_value) + min_scale

def cost_function(x,y,weight_x,weight_y):
    '''
    Cost function
    '''
    return (weight_x * x) + (weight_y * y)


def read_img(img_path, frame):
    full_path = os.path.join(img_path, str(frame).zfill(6) + '.jpg')
    return cv2.imread(full_path)

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
    
    bbox = tuple([img_bbox_left, img_bbox_top, img_bbox_w, img_bbox_h])
    return {'bbox':bbox, 'score':tm_conf}

#@profile
def track_templatematch(tracks, img_path, ttl_vtracking, window_size, tm_param):

    init_frame = min(tracks, key=lambda x: x['start_frame'])['start_frame']
    final_frame = max(tracks, key=lambda x: x['start_frame'])['start_frame']

    templates, front_tracks, rear_tracks = [], [], []

    _id = 1
    # tracking forwards
    for frame in range(init_frame, final_frame):
        img_read = False
        if templates:
            img = read_img(img_path, frame)
            img_read = True
            # update tracks at actual frame
            del_idxs = []
            for idx, tmplt in enumerate(templates):
                iou_track = next(track for track in tracks if track['id'] == tmplt['id'])
                max_frame = iou_track['start_frame'] + len(iou_track['bboxes']) + ttl_vtracking - 1
                if frame <= min(max_frame, final_frame):
                    # template matching
                    template_match = template_matching(img,tmplt['template'],tmplt['bbox'],window_size,tm_param)
                    rear_tracks.append({'frame':frame,'id':tmplt['id'],'bbox':template_match['bbox'],'score':template_match['score']})
                    # update template
                    tmplt['template'] = get_template(img,template_match['bbox'])
                    tmplt['bbox'] = template_match['bbox']
                else:
                    del_idxs.append(idx)
            # delete finished kcf trackers
            for del_idx in sorted(del_idxs, reverse=True):
                del templates[del_idx]
        
        for track in tracks:

            if frame == init_frame:
                track.update({'id':_id})
                _id += 1
            # initialization
            if track['start_frame'] + len(track['bboxes']) -1 == frame:
                if not img_read:
                    img = read_img(img_path, frame)
                    img_read = True
                template = get_template(img, track['bboxes'][-1])
                templates.append({'template':template, 'bbox':track['bboxes'][-1],'id':track['id']})
    # tracking backwards
    templates = []

    for frame in range(final_frame, init_frame, -1):
        img_read = False
        if templates:
            img = read_img(img_path, frame)
            img_read = True
            # update tracks at actual frame
            del_idxs = []
            for idx, tmplt in enumerate(templates):
                iou_track = next(track for track in tracks if track['id'] == tmplt['id'])
                min_frame = iou_track['start_frame'] - ttl_vtracking - 1
                if frame > max(min_frame, init_frame):
                    # template matching
                    template_match = template_matching(img,tmplt['template'],tmplt['bbox'],1,3)
                    front_tracks.append({'frame':frame,'id':tmplt['id'],'bbox':template_match['bbox'],'score':template_match['score']})
                    # update template
                    tmplt['template'] = get_template(img,template_match['bbox'])
                    tmplt['bbox'] = template_match['bbox']
                else:
                    del_idxs.append(idx)
            # delete finished kcf trackers
            for del_idx in sorted(del_idxs, reverse=True):
                del templates[del_idx]
        
        for track in tracks:
            # initialization
            if track['start_frame'] == frame:
                if not img_read:
                    img = read_img(img_path, frame)
                    img_read = True
                template = get_template(img, track['bboxes'][0])
                templates.append({'template':template, 'bbox':track['bboxes'][0],'id':track['id']})

    return tracks, front_tracks, rear_tracks
#@profile
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
                    if frame <= math.ceil((kcf_last_frame + kcf_start_frame) / 2):
                        kcf_bbox = next(bbox for bbox in track if bbox['frame'] == frame)
                    else:
                        kcf_bbox = next(bbox for bbox in front_tracks if bbox['frame'] == frame and bbox['id'] == best_assignment['id'])
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
            else:
                ii += 1
        else:
            ii += 1
    return main_tracks


def merge_tm(main_tracks, front_tracks, rear_tracks, sigma_iou_merge):
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
                    if frame <= math.ceil((kcf_last_frame + kcf_start_frame) / 2):
                        kcf_bbox = next(bbox for bbox in track if bbox['frame'] == frame)
                    else:
                        kcf_bbox = next(bbox for bbox in front_tracks if bbox['frame'] == frame and bbox['id'] == best_assignment['id'])
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
            else:
                ii += 1
        else:
            ii += 1
    return main_tracks