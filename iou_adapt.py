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
from util import load_mot, save_to_csv

# Fuer jede Sequenz eine Ordnerstruktur erstellen
def main(args):
    with open(args.seqmap) as fd:
        seqs = [line.rstrip('\n') for line in fd]

    for idx, seq in enumerate(seqs):
        if seq == "name" or seq == "":
            continue
        else:
            sigma_l = -0.5  # default -0.5
            sigma_h = 0.5   # default 0.5
            sigma_ext_iou = 0.4 # default 0.4
            t_min = 4       # default 4

            # uncomment line below if f4k2013 data is used
            # det_path = os.path.join(args.benchmark_dir,seq+"det.txt")
            # motchallenge data used
            det_path = os.path.join(args.benchmark_dir,seq,"det","det.txt")
            img_dir = os.path.join(args.benchmark_dir,seq,"img1")
            out_path = os.path.join(args.res_dir,seq+"_res.txt")
            detections = load_mot(det_path)

            start = time()
            # adapt line below
            tracks = track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min)
            end = time()

            num_frames = len(detections)
            print("finished " + seq + " at " + str(int(num_frames / (end - start))) + " fps!")

            save_to_csv(out_path, tracks)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extended IOU Tracker")
    parser.add_argument('-m', '--seqmap', type=str, required=True,
                        help="full path to the seqmap file to evaluate")
    parser.add_argument('-o', '--res_dir', type=str, required=True,
                        help="path to the results directory")
    parser.add_argument('-b', '--benchmark_dir', type=str, required=True,
                        help="path to the sequence directory")
    parser.add_argument('-w', '--weight_iou', type=str, required=False, default=0.7
                        help="weighting between iou and template matching")


    args = parser.parse_args()
    main(args)
