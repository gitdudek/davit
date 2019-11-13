import kcftracker
import os
import numpy as np
import cv2

dir = "/Volumes/Transcend/MOT17/train/MOT17-05-SDP/img1"
num_files =len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
inittracking = True
ontracking = False
res_tracker1 = []
res_tracker2 = []

for ii in range(num_files):

    frame_name = os.path.join(dir, str(ii+1).zfill(6)+".jpg")
    frame =  cv2.imread(frame_name)

    if inittracking:
        tracker1 = kcftracker.KCFTracker(False, True, True)  # hog, fixed_window, multiscale
        tracker2 = kcftracker.KCFTracker(False, True, True)  # hog, fixed_window, multiscale
        
        bbox1 = [17,150,77,191]
        bbox2 = [110,150,77,191]
        #bbox2 = [20,136,69,190]

        tracker1.init(bbox1, frame)
        tracker2.init(bbox2, frame)

        res_tracker1.append(bbox1)
        res_tracker2.append(bbox2)

        inittracking = False
        ontracking = True
    elif ontracking:

        res1 = tracker1.update(frame)
        res2 = tracker2.update(frame)

        res_tracker1.append([int(jj) for jj in res1])
        res_tracker2.append([int(kk) for kk in res2])

    cv2.rectangle(frame, (res_tracker1[-1][0], res_tracker1[-1][1]), (res_tracker1[-1][0]+res_tracker1[-1][2],res_tracker1[-1][1]+res_tracker1[-1][3]), (0,255,0),1)
    cv2.rectangle(frame, (res_tracker2[-1][0], res_tracker2[-1][1]), (res_tracker2[-1][0]+res_tracker2[-1][2],res_tracker2[-1][1]+res_tracker2[-1][3]), (0,0,255),1)

    cv2.imshow('tracking', frame)
    c = cv2.waitKey(30) & 0xFF
    if c==27 or c==ord('q'):
        break
        





