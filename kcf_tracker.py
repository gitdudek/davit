#import KCF.kcftracker as kcftracker
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

#@profile
def track_kcf(tracks, img_path, ttl_vtracking):
    """
    Implementation of the KCF-Tracker to extend the IOU results by visual informations

    Input parameters:
    tracks = Tracking results generated by the IOU-Tracker [frame, id, bb_left, bb_top, bb_width, bb_height]
    img_path = Path containing the images
    ttl_vtracking = Maximum length of frames for visual track

    Output parameters:
    tracks_iou_ext = Updated iou tracks with id's
    front_tracks = Front parts of the kcf tracks
    rear_tracks = Rear parts of the kcf tracks
    """
    init_frame = min(tracks, key=lambda x: x['start_frame'])['start_frame']
    final_frame = max(tracks, key=lambda x: x['start_frame'])['start_frame']
    front_tracks, rear_tracks = [], []
    _id = 1

    for track in tracks:
        track.update({'id':_id})
        # tracking backwards
        if track['start_frame'] > init_frame:
            # initialization
            tracker_bw = kcftracker.KCFTracker(False, True, True)
            frame_name = os.path.join(img_path, str(track['start_frame']).zfill(6) + '.jpg')
            img = cv2.imread(frame_name)
            tracker_bw.init(track['bboxes'][0], img)
            
            # iterate backwards through frames
            count_frames = 0
            for frame in range(track['start_frame'] - 1, track['start_frame'] - 1 - ttl_vtracking, -1):
                frame_name = os.path.join(img_path, str(frame).zfill(6) + '.jpg')
                img = cv2.imread(frame_name)
                tracking_update = tuple(tracker_bw.update(img))
                front_tracks.append({'frame':frame,'id':_id,'bbox':tracking_update})

                if frame <= 1 or count_frames >= ttl_vtracking - 1:
                    break

                count_frames += 1

        # tracking forwards
        last_frame = track['start_frame'] + len(track['bboxes']) - 1
        if last_frame < final_frame:
            # initialization
            tracker_fw = kcftracker.KCFTracker(False, True, True)
            frame_name = os.path.join(img_path, str(last_frame).zfill(6) + '.jpg')
            img = cv2.imread(frame_name)
            tracker_fw.init(track['bboxes'][-1], img)
            
            # iterate forwards through frames
            count_frames = 0
            for frame in range(last_frame+1, last_frame + ttl_vtracking + 1):
                frame_name = os.path.join(img_path, str(frame).zfill(6) + '.jpg')
                img = cv2.imread(frame_name)
                tracking_update = tuple(tracker_fw.update(img))
                rear_tracks.append({'frame':frame,'id':_id,'bbox':tracking_update})

                if frame >= final_frame or count_frames >= ttl_vtracking - 1:
                    break

                count_frames += 1
        _id += 1

    return tracks, front_tracks, rear_tracks

def read_img(img_path, frame):
    '''
    Return image
    '''
    frame_name = os.path.join(img_path, str(frame).zfill(6) + '.jpg')
    img = cv2.imread(frame_name)
    return img

#@profile
def track_kcf_2(tracks, img_path, ttl_vtracking):
    # TBD: Kopiere zu beginn tracks in eine neue Liste und lösche jeden Track für den ein kcf gestartet wurde!
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

### testing ###

'''import pandas as pd
from util import save_to_csv

directory = '/Volumes/Transcend/MOT17/train_adapt/MOT17-05-SDP/gt/gt.txt'
tracks = pd.DataFrame.from_csv(directory, header=None,index_col=False)
tracks_dict = {'bboxes':[],'max_score':0.876,'start_frame':6}
tracks_dict2 = {'bboxes':[],'max_score':0.876,'start_frame':5}
tracks_dict3 = {'bboxes':[],'max_score':0.876,'start_frame':25}

for ii in range(5,10,1):
    tracks_dict['bboxes'].append(tracks.iloc[ii:ii+1,[2,3,4,5]].values.tolist())

for ii in range(385,390,1):
    tracks_dict2['bboxes'].append(tracks.iloc[ii:ii+1,[2,3,4,5]].values.tolist())

for ii in range(737,757,1):
    tracks_dict3['bboxes'].append(tracks.iloc[ii:ii+1,[2,3,4,5]].values.tolist())

for jj in range(len(tracks_dict['bboxes'])):
    tracks_dict['bboxes'][jj] = tuple(tracks_dict['bboxes'][jj][0])
    tracks_dict2['bboxes'][jj] = tuple(tracks_dict2['bboxes'][jj][0])

for jj in range(len(tracks_dict3['bboxes'])):
    tracks_dict3['bboxes'][jj] = tuple(tracks_dict3['bboxes'][jj][0])

#print(tracks_dict['bboxes'])
tracks_list = [tracks_dict] + [tracks_dict2] + [tracks_dict3]


res_main, res_front, res_rear = track_kcf(tracks_list, "/Volumes/Transcend/MOT17/train_adapt/MOT17-05-SDP/img1", 10)
#print(tracking_results)
save_to_csv('/Volumes/Transcend/MOT17/kcf_iou_out/res.txt',tracking_results)'''