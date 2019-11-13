# ---------------------------------------------------------
# DAVIT Tracker
# Written by Christoph Dudek
# ---------------------------------------------------------

import KCF.kcftracker as kcftracker
import os
import cv2

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


def read_img(img_path, frame):
    '''
    Return image
    '''
    frame_name = os.path.join(img_path, str(frame).zfill(6) + '.jpg')
    img = cv2.imread(frame_name)
    return img

#@profile
def track_kcf(tracks, img_path, ttl_vtracking):
    
    init_frame = min(tracks, key=lambda x: x['start_frame'])['start_frame']
    final_frame = max(tracks, key=lambda x: x['start_frame'])['start_frame']

    kcf, front_tracks, rear_tracks = [], [], []

    _id = 1
    # tracking forwards
    for frame in range(init_frame, final_frame):
        img_read = False
        if kcf:
            img = read_img(img_path, frame)
            img_read = True
            # update tracks at actual frame
            del_idxs = []
            for idx, kcf_tracker in enumerate(kcf):
                iou_track = next(track for track in tracks if track['id'] == kcf_tracker['id'])
                max_frame = iou_track['start_frame'] + len(iou_track['bboxes']) + ttl_vtracking - 1
                if frame <= min(max_frame, final_frame):
                    track_update = tuple(kcf_tracker['kcf'].update(img))
                    rear_tracks.append({'frame':frame,'id':kcf_tracker['id'],'bbox':track_update})
                else:
                    del_idxs.append(idx)
            # delete finished kcf trackers
            for del_idx in sorted(del_idxs, reverse=True):
                del kcf[del_idx]
        
        for track in tracks:

            if frame == init_frame:
                track.update({'id':_id})
                _id += 1
            # initialization
            if track['start_frame'] + len(track['bboxes']) -1 == frame:
                if not img_read:
                    img = read_img(img_path, frame)
                    img_read = True
                kcf.append({'kcf':kcftracker.KCFTracker(False,True,True),'id':track['id']})
                kcf[-1]['kcf'].init(track['bboxes'][-1],img)

    # tracking backwards
    kcf = []

    for frame in range(final_frame, init_frame, -1):
        img_read = False
        if kcf:
            img = read_img(img_path, frame)
            img_read = True
            # update tracks at actual frame
            del_idxs = []
            for idx, kcf_tracker in enumerate(kcf):
                iou_track = next(track for track in tracks if track['id'] == kcf_tracker['id'])
                min_frame = iou_track['start_frame'] - ttl_vtracking - 1
                if frame > max(min_frame, init_frame):
                    track_update = tuple(kcf_tracker['kcf'].update(img))
                    front_tracks.append({'frame':frame,'id':kcf_tracker['id'],'bbox':track_update})
                else:
                    del_idxs.append(idx)
            # delete finished kcf trackers
            for del_idx in sorted(del_idxs, reverse=True):
                del kcf[del_idx]
        
        for track in tracks:
            # initialization
            if track['start_frame'] == frame:
                if not img_read:
                    img = read_img(img_path, frame)
                    img_read = True
                kcf.append({'kcf':kcftracker.KCFTracker(False,True,True),'id':track['id']})
                kcf[-1]['kcf'].init(track['bboxes'][0],img)
            

    

    return tracks, front_tracks, rear_tracks