#!/usr/bin/env python

# ---------------------------------------------------------
# IOU Tracker
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Licensed under The MIT License [see LICENSE for details]
# Written by Erik Bochinski
# ---------------------------------------------------------

from time import time
import argparse
import os
from iou_tracker import track_iou
from kcf_tracker import track_kcf, track_kcf_2
from util import load_mot, save_to_csv, merge, track_templatematch

# Fuer jede Sequenz eine Ordnerstruktur erstellen
def main(args):

    ttl_vtracking_candidates = [1,2,3,4,5,6,7,8,9,10]
    sigma_iou_merge_candidates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    window_size_candidates = [0.2,0.4,0.6,0.8,1]
    tm_param_candidates = [3]
    counter = 1

    for candidate_ttl in ttl_vtracking_candidates:
        for candidate_sigma in sigma_iou_merge_candidates:
            for candidate_window_size in window_size_candidates:
                
                print('Evaluating combination ' + str(counter) + ' of '  + str(len(ttl_vtracking_candidates)+len(sigma_iou_merge_candidates)+len(window_size_candidates)))
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

                        # uncomment line below if f4k2013 data is used
                        # det_path = os.path.join(args.benchmark_dir,seq+"det.txt")
                        # motchallenge data used
                        det_path = os.path.join(args.benchmark_dir,seq,"det","det.txt")
                        img_path = os.path.join(args.benchmark_dir,seq,"img1")
                        out_dir = os.path.join(args.res_dir,'kcf_vis_len_'+str(candidate_ttl)+'_sigma_iou_merge_'+str(candidate_sigma)+'_window_size_'+str(candidate_window_size))
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        out_path = os.path.join(out_dir,seq+".txt")
                        detections = load_mot(det_path)

                        start = time()
                        tracks_iou = track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min)
                        tracks_iou_ext, tracks_kcf_front, tracks_kcf_rear = track_templatematch(tracks_iou, img_path, candidate_ttl, candidate_window_size, tm_param_candidates)
                        tracks_merged = merge(tracks_iou_ext, tracks_kcf_front, tracks_kcf_rear, candidate_sigma)
                        end = time()

                        num_frames = len(detections)
                        print("finished " + seq + " at " + str(int(num_frames / (end - start))) + " fps!")

                        save_to_csv(out_path, tracks_merged)
                counter += 1 

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
