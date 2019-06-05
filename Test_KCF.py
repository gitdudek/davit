import KCF.kcftracker as kcftracker
import cv2
import os

trackers, tracks = [], []

for i in range(1,10):
    frame_name = os.path.join('/Volumes/Transcend/MOT17/train/MOT17-05-SDP/img1', str(i).zfill(6) + '.jpg')
    img = cv2.imread(frame_name)
    if i == 1:
        for j in range(5):
            trackers.append(kcftracker.KCFTracker(False,True,True))
            trackers[j].init((10,20,30,i),img)
        else:
            for k in range(5):
                bbox = trackers[k].update(img)
                print(type(bbox))

print(trackers)
