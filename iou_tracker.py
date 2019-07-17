# ---------------------------------------------------------
# IOU Tracker
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Licensed under The MIT License [see LICENSE for details]
# Written by Erik Bochinski
# ---------------------------------------------------------
import numpy as np
from util import *
from time import time
from scipy.stats import hmean
from sklearn.utils.linear_assignment_ import linear_assignment



def track_iou2(detections, sigma_l, sigma_h, sigma_iou, t_min):
    '''
    Verbesserung:
    1) Nur Detections für Richtungsvergleich verwenden, deren IOU größer null mit dem aktuellen Track sind
    2) Mittelwert zur Bestimmung der Richtung ist kein gutes Maß! (Wenn Richtung 0 und 180, dann Mean Richtung 180)
    3) Richtung evtl. erst ab Länge von 3 mit aufnehmen (Wenn Richtung 0 und 359, dann mittlere Richtung 180)
    4) Geschwindigkeit der Objekte mit aufnehmen
    5) Ausreißer der Directions filtern oder schwächer wichten!
    6) Postprocessing, indem alle Tracks verworfen werden, deren Gesamt-IOU sich über einen längeren Zeitraum
       mit einem anderen Track überschneidet! (Behalten des längeren Tracks
       oder des Tracks mit der höheren Confidence)
    '''
    # Filter detections for certain area around the tracked object when assigning the direction!
    weight_iou = 0.7
    weight_direction = 1 - weight_iou
    sigma_cost = 0.7
    frames_mean = 20

    tracks_active = []
    tracks_finished = []

    for frame_num, detections_frame in enumerate(detections, start=1):
        # apply low threshold to detections
        dets = [det for det in detections_frame if det['score'] >= sigma_l]

        updated_tracks = []

        if len(dets) > 0:
            
            cost_matrix = np.zeros((len(tracks_active),len(dets)),dtype=np.float32)
            
            for t, track in enumerate(tracks_active):
                
                len_track = len(track['bboxes'])

                if len_track >= 2:
                    center_bbox1 = get_center(track['bboxes'][-2])
                    center_bbox2 = get_center(track['bboxes'][-1])
                    #previous_direction = get_angle(center_bbox1, center_bbox2)
                    if previous_direction == 0:
                        previous_direction += 1e-10
                    if len_track == 2:
                        track['directions'] = [previous_direction]
                    else:
                        track['directions'].append(previous_direction)
                    # calculate the mean direction of a maximum amount of the 10 last frames
                    mean_direction = hmean(track['directions'][-min(len_track-1,frames_mean):])
                    #mean_direction = np.mean(track['directions'][-min(len_track-1,frames_mean):])

                #if len(dets) > 0:
                for d, det in enumerate(dets):
                    
                    _iou = iou(track['bboxes'][-1], det['bbox'])
                    scaled_iou = scale(_iou, min_value=1, max_value=0, min_scale=0, max_scale=1)

                    if len_track >= 20:
                        center_det = get_center(det['bbox'])
                        direction = get_angle(center_bbox2, center_det)
                        if direction > 180:
                            direction = abs(direction-360)
                        deviation = get_deviation(mean_direction, direction)
                        scaled_direction = scale(deviation, min_value=0, max_value=180, min_scale=0, max_scale=1)
                        
                        if _iou > 0:
                            costs = cost_function(scaled_iou, scaled_direction, weight_iou, weight_direction)
                        else:
                            costs = 1
                        cost_matrix[t,d] = costs

                        '''if costs <= sigma_cost:
                            cost_matrix[t,d] = costs
                        else:
                            cost_matrix[t,d] = 1'''
                    else:
                        if _iou >= sigma_iou:
                            cost_matrix[t,d] = scaled_iou
                        else:
                            cost_matrix[t,d] = 1
            
            assigned_det_idxs = []
            if tracks_active: 
                # assignment
                matched_indices = linear_assignment(cost_matrix)

                for match in matched_indices:
                    
                    if cost_matrix[match[0],match[1]] <= sigma_cost:
                        track = tracks_active[match[0]]
                        detection = dets[match[1]]
                        track['bboxes'].append(detection['bbox'])
                        track['max_score'] = max(track['max_score'], detection['score'])

                        updated_tracks.append(track)
                        assigned_det_idxs.append(match[1])

                for det_idx in sorted(assigned_det_idxs, reverse=True):
                    del dets[det_idx]

        if len(updated_tracks) == 0 or len(tracks_active) != len(updated_tracks):
            for active_track in tracks_active:
                if not active_track in updated_tracks:
                    # finish track when the conditions are met
                    if active_track['max_score'] >= sigma_h and len(active_track['bboxes']) >= t_min:
                        tracks_finished.append(active_track)

        # create new tracks
        new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_num} for det in dets]
        tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active
                        if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]

    return tracks_finished


def track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min):
    """
    Simple IOU based tracker.
    See "High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora" for
    more information.

    Args:
         detections (list): list of detections per frame, usually generated by util.load_mot
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.

    Returns:
        list: list of tracks.
    """

    tracks_active = []
    tracks_finished = []

    for frame_num, detections_frame in enumerate(detections, start=1):
        # apply low threshold to detections
        dets = [det for det in detections_frame if det['score'] >= sigma_l]

        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # get det with highest iou
                best_match = max(dets, key=lambda x: iou(track['bboxes'][-1], x['bbox']))
                if iou(track['bboxes'][-1], best_match['bbox']) >= sigma_iou:
                    track['bboxes'].append(best_match['bbox'])
                    track['max_score'] = max(track['max_score'], best_match['score'])

                    updated_tracks.append(track)

                    # remove from best matching detection from detections
                    del dets[dets.index(best_match)]

            # if track was not updated
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                # finish track when the conditions are met
                if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min:
                    tracks_finished.append(track)

        # create new tracks
        new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_num} for det in dets]
        tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active
                        if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]

    return tracks_finished


def track_iou_matlab_wrapper(detections, sigma_l, sigma_h, sigma_iou, t_min):
    """
    Matlab wrapper of the iou tracker for the detrac evaluation toolkit.

    Args:
         detections (numpy.array): numpy array of detections, usually supplied by run_tracker.m
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.

    Returns:
        float: speed in frames per second.
        list: list of tracks.
    """

    detections = detections.reshape((7, -1)).transpose()
    dets = load_mot(detections)
    start = time()
    tracks = track_iou(dets, sigma_l, sigma_h, sigma_iou, t_min)
    end = time()

    id_ = 1
    out = []
    for track in tracks:
        for i, bbox in enumerate(track['bboxes']):
            out += [float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]),
                    float(track['start_frame'] + i), float(id_)]
        id_ += 1

    num_frames = len(dets)
    speed = num_frames / (end - start)

    return speed, out
