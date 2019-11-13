#!/usr/bin/env python

# ---------------------------------------------------------
# DAVIT Tracker
# Written by Christoph Dudek
# ---------------------------------------------------------

from time import time
import argparse
import os
from iou_tracker import track_iou
from util import load_mot, save_to_csv, merge, track_templatematch, detections_preprocessing


def main(args):

    time_iou = 0
    time_ext = 0
    time_merge = 0
    time_ttl = 0

    with open(args.seqmap) as fd:
        seqs = [line.rstrip('\n') for line in fd]

    for idx, seq in enumerate(seqs):
        if seq == "name" or seq == "":
            continue
        else:
            if "DPM" in seq:
                sigma_l = -0.5
                sigma_h = 0.5
                sigma_iou = 0.5
                t_min = 4
            elif "FRCNN" in seq:
                sigma_l = 0.0
                sigma_h = 0.9
                sigma_iou = 0.4
                t_min = 3
            elif "SDP" in seq:
                sigma_l = 0.4
                sigma_h = 0.5
                sigma_iou = 0.3
                t_min = 2
            else:
                sigma_l = 0.4       # default -0.5
                sigma_h = 0.5       # default 0.5
                sigma_iou = 0.2     # default 0.4
                t_min = 2           # default 4

            ttl_vtracking = 8 # maximum length of visual track (amount of framess)
            sigma_iou_merge = 0.2

            # uncomment line below if f4k2013 data is used
            # det_path = os.path.join(args.benchmark_dir,seq+"det.txt")
            # motchallenge data used
            det_path = os.path.join(args.benchmark_dir,seq,"det","det.txt")
            img_path = os.path.join(args.benchmark_dir,seq,"img1")
            out_path = os.path.join(args.res_dir,seq+".txt")
            detections = load_mot(det_path)

            detections = detections_preprocessing(detections, iou_filter=0.92)


            start = time()
            tracks_iou = track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min)
            end_iou = time()
            tracks_iou_ext, tracks_kcf_front, tracks_kcf_rear = track_templatematch(tracks_iou, img_path, ttl_vtracking, 0.5, 3)
            end_ext = time()
            tracks_merged = merge(tracks_iou_ext, tracks_kcf_front, tracks_kcf_rear, sigma_iou_merge)
            end = time()

            num_frames = len(detections)
            print("finished " + seq + " at " + str(int(num_frames / (end - start))) + " fps!")

            save_to_csv(out_path, tracks_merged)

            time_iou += (end_iou-start)
            time_ext += (end_ext-end_iou)
            time_merge += (end-end_ext)
            time_ttl += (end-start)

    print("Time IOU: " + str(time_iou))
    print('Time VOT: ' + str(time_ext))
    print('Time Merge:' + str(time_merge))
    print('Time total: ' + str(time_ttl))            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extended IOU Tracker")
    parser.add_argument('-m', '--seqmap', type=str, required=True,
                        help="full path to the seqmap file to evaluate")
    parser.add_argument('-o', '--res_dir', type=str, required=True,
                        help="path to the results directory")
    parser.add_argument('-b', '--benchmark_dir', type=str, required=True,
                        help="path to the sequence directory")


    args = parser.parse_args()
    main(args)
