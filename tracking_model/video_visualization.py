# --------------------------------------------------------
# Video Visualization
# Licensed under The MIT License
# Written by limengyao(mengyao.lmy@alibaba-inc.com)
# --------------------------------------------------------
#!/usr/bin/python

import cv2
import numpy as np
import logging
import os
import argparse
import re
import time

from tracker import Tracker
from statistic.utils import rect_2_cxy_wh, cxy_wh_2_rect, corner_to_rect, get_none_rotation_rect, \
    compute_success_overlap


def visualize_result(frame, location, gt_rect=None):
    if len(location) == 8:
        cv2.polylines(frame, [location.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
    else:
        location = [int(l) for l in location]  #
        cv2.rectangle(frame, (location[0], location[1]),
                      (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 3)

    if gt_rect is not None:
        if len(gt_rect) == 8:
            cv2.polylines(frame, [np.array(gt_rect, np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        else:
            cv2.rectangle(frame, (gt_rect[0], gt_rect[1]), (gt_rect[0] + gt_rect[2], gt_rect[1] + gt_rect[3]), (0, 255, 0), 3)
    return frame


def eval_video(gt_rects, interval, tracker):
    if not os.path.isfile(args.video_path):
        logging.error('Video not exist!')
        exit()
    cap = cv2.VideoCapture(args.video_path)

    toc, regions, scores = 0, [], []
    gt = get_none_rotation_rect(np.array(gt_rects))
    gt = corner_to_rect(gt)
    f = 0

    ret, frame = cap.read()
    if frame is None:
        exit()

    if args.result_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.result_video, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    while True:
        tic = cv2.getTickCount()

        if (interval == -1 and f ==0) or (interval != -1 and f % interval == 0):
            target_pos, target_sz = rect_2_cxy_wh(gt[f])
            tracker.init_tracker(frame, target_pos, target_sz)
            location = cxy_wh_2_rect(tracker.state['target_pos'], tracker.state['target_sz'])
            scores.append(1.0)
            regions.append(location)
        if (interval == -1 and f !=0) or f % interval != 0:  # tracking
            location = tracker.update_tracker(frame)
            scores.append(tracker.state['score'])
            regions.append(location)

        toc += cv2.getTickCount() - tic

        if args.visualization and f >= 0:  # visualization
            frame = visualize_result(frame, location, gt[f])
            cv2.putText(frame, '{}, Score:{:.2f}'.format(f, scores[f]), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('Tracking Result', cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break

            #Save video
            if args.result_video != None:
                writer.write(frame)

        f += 1
        ret, frame = cap.read()
        if frame is None:
            break
    if args.result_file is not None:
        with open(args.result_file, "w") as fin:
            for x in regions:
                fin.write(','.join([str(i) for i in x])+'\n')

    toc /= cv2.getTickFrequency()
    return f/toc, regions


def test_video(tracker):
    start_time = time.time()
    if not os.path.isfile(args.video_path):
        logging.error('Video not exist!')
        exit()
    cap = cv2.VideoCapture(args.video_path)

    ret, frame = cap.read()
    if frame is None:
        exit()

    if args.result_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter()
        success = writer.open(args.result_video, fourcc, 20.0, (frame.shape[1], frame.shape[0]), True)

    #crop_win_name = 'Test Video'
    #cv2.namedWindow(crop_win_name, cv2.WND_PROP_FULLSCREEN)

    try:
        #init_rect = cv2.selectROI(crop_win_name, frame, False, False)
        init_rect = (405,335,399,442)
        x, y, w, h = init_rect
        gt_rect = [x, y, w, h]
    except:
        exit()

    toc, regions, scores = 0, [], []
    f = 0

    while True:
        tic = cv2.getTickCount()

        if f == 0:
            target_pos, target_sz = rect_2_cxy_wh(gt_rect)
            tracker.init_tracker(frame, target_pos, target_sz)
            location = cxy_wh_2_rect(target_pos, target_sz)
            regions.append(location)
            scores.append(1.0)
        if f > 0:  # tracking
            location = tracker.update_tracker(frame)
            scores.append(tracker.state['score'])
            regions.append(location)

        toc += cv2.getTickCount() - tic

        if args.visualization and f >= 0:  # visualization
            if f == 0:
                frame = visualize_result(frame, location, gt_rect)
            if f > 0:
                frame = visualize_result(frame, location)
            cv2.putText(frame, '{}, Score:{:.2f}'.format(f, scores[f]), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow(crop_win_name, frame)
            # cv2.imwrite(os.path.join('face_kcf_notrain_rawpixel', 'out_{:05d}.jpg'.format(f)), cv2.resize(frame, (180, 360)))
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break

        #Save video
        if args.result_video != None:
            writer.write(frame)

        f += 1
        ret, frame = cap.read()
        if frame is None:
            break
    if args.result_video != None:
        writer.release()
        print('Video Process Done! Result: {}'.format(success))
    if args.result_file is not None:
        with open(args.result_file, "w") as fin:
            for x in regions:
                fin.write(','.join([str(i) for i in x])+'\n')

    # close all open windows
    cv2.destroyAllWindows()
    print(time.time()-start_time)

    toc /= cv2.getTickFrequency()
    return f/toc, regions



def main():
    global args, device

    parser = argparse.ArgumentParser(description='Tracking Video Visualization')
    parser.add_argument('--video_path', default='test_data/ohaqlzfnuv.mp4', help='video path')
    parser.add_argument('--ground_truth', help='ground truth file path')
    parser.add_argument('--model', default='models/SiamRPNOTB.model', help='model path')
    parser.add_argument('--net_name', type=str, default='SiamRPNotb', help='network, SiamRPNbatchOTBMobile, SiamRPNotb')
    parser.add_argument('--interval', type=int, default=-1, help='initialize interval')
    parser.add_argument('--mode', type=str, default='test', help='[test, eval]')
    parser.add_argument('--result_video', help='result video')
    parser.add_argument('--result_file', default='saved_data/result.txt', help='result file')
    parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                        help='whether visualize result')

    args = parser.parse_args()
    args.visualization = False

    tracker = Tracker(0, args.net_name, args.model)

    if args.mode == 'eval' and args.ground_truth is not None:
        with open(args.ground_truth) as f:
            labels = f.readlines()
        gt_rects = []
        for label in labels:
            label = re.split(';|,|\t| |, |; ', label.strip('\n'))
            label = list(map(int, label))
            gt_rects.append(label)

        # Todo: multi video test
        thresholds_overlap = np.arange(0, 1.05, 0.05)
        success_overlap = np.zeros(len(thresholds_overlap))
        fps, regions = eval_video(gt_rects, args.interval, tracker)
        gt_rects = get_none_rotation_rect(np.array(gt_rects))
        gt_rects = corner_to_rect(gt_rects)
        success_overlap = compute_success_overlap(gt_rects, np.array(regions))
        auc = success_overlap.mean()

        logging.info('Mean Running AP:{:.4f} Mean Running Speed {:.1f}fps'.format(auc, fps))

    elif args.mode == 'test':
        assert(args.interval == -1)
        fps, regions = test_video(tracker)
        print(fps)


if __name__ == '__main__':
    logging.basicConfig(format='[%(process)d] %(asctime)s: %(message)s', level=logging.INFO)
    main()


