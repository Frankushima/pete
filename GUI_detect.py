import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

import logic_tools
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, bbox_iou
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import threading
import time
import tkinter as tk
from CustomScrollBar import ScrollBar
from shared import *
from Step import Step
from PIL import Image, ImageTk

from logic_tools import *
import queue
import math
import emoji

from collections import Counter

current_step = 0
procedure = []
gui = None
detect_ready = threading.Event()
cv_queue = queue.Queue()
sensor_queue = queue.Queue()
terminate = threading.Event()
sensor_in_use = threading.Event()

def detect(save_img=False):
    global gui, cv_queue
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        temp = [None, None]
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                hand_count, hand_det = logic_tools.find_hands(det)
                if hand_count == 2:
                    L_hand_det, R_hand_det = logic_tools.RL_hands(hand_det)

                cv_queue.put(det)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        if cls == 4 and hand_count == 2:  # 4 is for hand (might need to change in future)
                            xyxy_list_tensor = torch.stack(xyxy)  # convert type for comparison

                            if torch.all(xyxy_list_tensor == L_hand_det[:4]):
                                label = f"Left Hand {conf:.2f}"
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                            elif torch.all(xyxy_list_tensor == R_hand_det[:4]):
                                label = f"Right Hand {conf:.2f}"
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)


                        else:
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            temp[i] = im0

            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        
        # allow GUI to set up properly before start (not needed on board)
        # detect_ready.set()

        if not detect_ready.is_set():
            detect_ready.set()
            time.sleep(1)

        # Stream results
        if view_img:
            # only stack videos on webcams two inputs
            if source.endswith('.txt'):
                final = cv2.vconcat(temp)
            else:
                final = temp[0]
            gui.set_frame(final)

            cv2.waitKey(1)  # 1 millisecond

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


sample = 0
dummy_sensor_data = process("./test_sensor_data/output.txt")
ewma_sd = 0  # ewma sensor data
sum_sd = 0
def sensor_detect():
    """
    Dummy sensor
    """
    global sensor_in_use, terminate, sensor_queue, sample, ewma_sd, sum_sd

    while not terminate.is_set():
        # if not using sensor tool, simply ignore data
        if not sensor_in_use.is_set():
            time.sleep(1)
        else:
            # print("sensor in use = ================================================")

            # Read in + a lil EWMA
            if sample == 0:
                ewma_sd = dummy_sensor_data[sample] * 1.4
            else:
                ewma_sd = 0.75 * ewma_sd + 0.25 * dummy_sensor_data[
                    sample % len(dummy_sensor_data)] * 1.4  # 1.4 for calibration purposes
            sum_sd += ewma_sd
            sample += 1

            # Matching camera and sensor via sampling rate (ceiling)
            if not camera_sensor_frame_match(x=sample): continue

            # Process (should also increase sampling rate since processing takes time?)
            is_rotating = ewma_sd > 3.5  # lmao

            data = {
                'rotating': is_rotating,
                'degrees': sum_sd,
                'num_rotations': int(sum_sd / 360),
            }

            sensor_queue.put(data)
            time.sleep(0.1)  # a bit delay to prevent thread dying


def decision_logic():
    global procedure, current_step, gui, cv_queue
    while True:  # prevent calling before initialization
        if gui is not None: break

    """
    Decision making frame goes here: we're simply calling validate() on the current step. Each step has its 
    own `validate()` method that is defined at initialization of the procedure. The crux of decision logic is 
    stored in each step, since each step has different criteria (unless there's some other way to implement it)

    (?) Methods that should be called here are 1) CV detect and 2) sensor detect. 

    :param data: a dict of CV data (bounding boxes) and sensor data. Or, evoke methods in main to get these data.
    """
    # data['CV'] = blah blah
    # data['sensor'] = blob blob
    # data['lala'] = wawa

    # TESTING ONLY: always validate to true after 10 seconds (lol)
    # if procedure[current_step].validate(data):
    #     gui.mark_step_done(DONE)

    # Currently validate function will not be used because of the code structure
    # Each step has its own validate function however, the class itself cannot modify the validate function
    # the validate method willb e implemented here hardcoded (at least for now...)

    # initialize make step 1 to IN_PROGRESS
    procedure[current_step].update_status(IN_PROGRESS)

    # build substeps for step 1
    gui.build_substeps(procedure[current_step])

    # So basically run the validators sequentially when we revert to make sure we restart the validation properly
    while not terminate.is_set():
        step1_validator()

        step2_validator()
        
        step3_validator()
        
        step4_validator()        

        step5_validator()

        step6_validator()

        step_runtime = step7_validator()
        print(f"Step 7 runtime={step_runtime} secs")

# ===================== Step Logics ============================
def step1_validator():
    # variables for trendline, must be initalize outside of steps
    # Sub3
    s3_prev_dist_R_Spindle = -math.inf
    s3_trend_R_Spindle = logic_tools.Trendline.INITIALIZE

    s3_prev_dist_L_Spindle = math.inf
    s3_trend_L_Spindle = logic_tools.Trendline.INITIALIZE

    # Sub5
    s5_prev_dist_Spindle = -math.inf
    s5_trend_Spindle = logic_tools.Trendline.INITIALIZE

    # Detections Expected: Left Hand, Right Hand, Spindle
    sub_conditions = [False for i in range(7)]
    while current_step == 0:
        data = cv_queue.get()
        num_class_detected = len(data)

        # SUB 0 : is there a hand?
        if not sub_conditions[0]:
            hand_count, hands_det = logic_tools.find_hands(data)
            if hand_count >= 1:
                gui.update_substep(0)
                sub_conditions[0] = True

        # SUB 1 : is there a spindle? (index = 7)
        if not sub_conditions[1] and sub_conditions[0] == True:
            spindle_count, spindle_det = logic_tools.find_class(data, SPINDLE)
            if spindle_count == 1:
                gui.update_substep(1)
                sub_conditions[1] = True

        # SUB 2 : are they overlapped? hand holding spindle?
        if not sub_conditions[2] and sub_conditions[1] == True:
            over_count = 0
            if num_class_detected > 1:
                over_count, over_det, over_dict = logic_tools.find_overlapping(data)
                if over_count == 1:
                    single_overlap_pair = over_det[0]
                    # if the overlapping is between spindle and hand
                    if (single_overlap_pair[0][5] == SPINDLE and single_overlap_pair[1][5] == HAND) \
                            or (single_overlap_pair[0][5] == HAND and single_overlap_pair[1][5] == SPINDLE):
                        gui.update_substep(2)
                        sub_conditions[2] = True

        # SUB 3 : leaving right hand + increasing left hand
        # TODO: Fix IOU not gonna work for varying size for bounding box
        # Using Euclidean Distane for now
        if not sub_conditions[3] and sub_conditions[2] == True:
            s3_curr_dist_R_Spindle = -1
            s3_curr_dist_L_Spindle = -1

            if num_class_detected > 1:
                hand_count, hands_det = logic_tools.find_hands(data)
                over_count, over_det, over_dict = logic_tools.find_overlapping(data)
                spindle_count, spindle_det = logic_tools.find_class(data, 7)

                if hand_count == 2 and spindle_count == 1:
                    L_hand_det, R_hand_det = logic_tools.RL_hands(hands_det)

                    spindle_center = logic_tools.get_box_center(*spindle_det[0][:4])

                    # Right Hand
                    R_hand_center = logic_tools.get_box_center(*R_hand_det[:4])
                    s3_curr_dist_R_Spindle = logic_tools.get_euclidean_distance(R_hand_center, spindle_center)

                    # Spindle Leaving Right Hand
                    if s3_curr_dist_R_Spindle > s3_prev_dist_R_Spindle:
                        s3_trend_R_Spindle = logic_tools.Trendline.INCREASING

                    elif s3_curr_dist_R_Spindle < s3_prev_dist_R_Spindle:
                        s3_trend_R_Spindle = logic_tools.Trendline.DECREASING

                    s3_prev_dist_R_Spindle = s3_curr_dist_R_Spindle

                    # Left Hand
                    L_hand_center = logic_tools.get_box_center(*L_hand_det[:4])
                    s3_curr_dist_L_Spindle = logic_tools.get_euclidean_distance(L_hand_center, spindle_center)

                    # Spindling Going to Left Hand
                    if s3_curr_dist_L_Spindle < s3_prev_dist_L_Spindle:
                        s3_trend_L_Spindle = logic_tools.Trendline.DECREASING

                    elif s3_curr_dist_L_Spindle > s3_prev_dist_L_Spindle:
                        s3_trend_L_Spindle = logic_tools.Trendline.INCREASING

                    s3_prev_dist_L_Spindle = s3_curr_dist_L_Spindle

                    # print(f"Left Trend: {s3_trend_L_Spindle} Right Trend: {s3_trend_R_Spindle}")
                if hand_count == 1 and spindle_count == 1 and s3_trend_R_Spindle == logic_tools.Trendline.INCREASING and s3_trend_L_Spindle == logic_tools.Trendline.DECREASING:
                    gui.update_substep(3)
                    sub_conditions[3] = True

        # SUB 4 : passed to left hand
        if not sub_conditions[4] and sub_conditions[3] == True:
            over_count = 0

            if num_class_detected > 1:
                over_count, over_det, over_dict = logic_tools.find_overlapping(data)
                if over_count == 1:
                    single_overlap_pair = over_det[0]
                    # if the overlapping is between spindle and hand
                    if ((single_overlap_pair[0][5] == SPINDLE and single_overlap_pair[1][5] ==
                         HAND) or
                            (single_overlap_pair[0][5] == HAND and single_overlap_pair[1][5] ==
                             SPINDLE)):
                        gui.update_substep(4)
                        sub_conditions[4] = True

        # SUB 5 : leaving left hand
        if not sub_conditions[5] and sub_conditions[4] == True:
            s5_curr_dist_Spindle = -1

            if num_class_detected > 1:
                hand_count, hands_det = logic_tools.find_hands(data)
                over_count, over_det, over_dict = logic_tools.find_overlapping(data)
                spindle_count, spindle_det = logic_tools.find_class(data, 7)

                if hand_count == 1 and spindle_count == 1:
                    spindle_center = logic_tools.get_box_center(*spindle_det[0][:4])
                    hand_center = logic_tools.get_box_center(*hands_det[0][:4])

                    s5_curr_dist_Spindle = logic_tools.get_euclidean_distance(hand_center, spindle_center)

                    # Spindle Leaving Right Hand
                    if s5_curr_dist_Spindle > s5_prev_dist_Spindle:
                        s5_trend_Spindle = logic_tools.Trendline.INCREASING

                    elif s5_curr_dist_Spindle < s5_prev_dist_Spindle:
                        s5_trend_Spindle = logic_tools.Trendline.DECREASING

                    s5_prev_dist_Spindle = s5_curr_dist_Spindle

            if num_class_detected == 1 and s5_trend_Spindle == logic_tools.Trendline.INCREASING:
                gui.update_substep(5)
                sub_conditions[5] = True

        # SUB 6 : no overlap spingle left behind
        if not sub_conditions[6] and sub_conditions[5] == True:
            spindle_count, spindle_det = logic_tools.find_class(data, SPINDLE)

            if spindle_count == 1 and num_class_detected == 1:
                gui.update_substep(6)
                sub_conditions[6] = True

        if all(sub_conditions):
            print("Step 1 Done")
            gui.mark_step_done(DONE)

        # print(f"Spindle: {spindle_count}, Hand: {hand_count}, Overlapping_Count: {over_count}, Overlapping_IOU: {iou}")

def step2_validator():
     # Detections Expected: Left Hand, Right Hand, DoubleFlatBottomBracket, (Spindle)
    # Both Hand hold Bracket
    # Single Hand Tighten
    sub_conditions = [False for i in range(9)]
    while current_step == 1:
        data = cv_queue.get()
        num_class_detected = len(data)

        # SUB 0 : is there a hand?
        if not sub_conditions[0]:
            hand_count, hands_det = logic_tools.find_hands(data)
            if hand_count >= 1:
                gui.update_substep(0)
                sub_conditions[0] = True

        # SUB 1 : is there a doubleflatbottombracket?
        if not sub_conditions[1] and sub_conditions[0] == True:
            double_flat_bb_count, _ = logic_tools.find_class(data, DOUBLEFLATS_BOTTOM_BRACKET)
            if double_flat_bb_count == 1:
                gui.update_substep(1)
                sub_conditions[1] = True

        # SUB 2 : is there a spindle?
        if not sub_conditions[2] and sub_conditions[1] == True:
            spindle_count, _ = logic_tools.find_class(data, SPINDLE)
            if spindle_count == 1:
                gui.update_substep(2)
                sub_conditions[2] = True

        # SUB 3: hand holding doubleflat bottom bracket
        if not sub_conditions[3] and sub_conditions[2] == True:
            hand_count, hand_det = logic_tools.find_class(data, HAND)
            double_flat_bb_count, double_flat_bb_det = logic_tools.find_class(data, DOUBLEFLATS_BOTTOM_BRACKET)

            for hand in hand_det:
                overlapping, _ = logic_tools.is_overlapping(hand, double_flat_bb_det[0])

            if overlapping:
                gui.update_substep(3)
                sub_conditions[3] = True

        # SUB 4: bottom bracket completely covering spindle
        if not sub_conditions[4] and sub_conditions[3] == True:
            double_flat_bb_count, double_flat_bb_det = logic_tools.find_class(data, DOUBLEFLATS_BOTTOM_BRACKET)
            spindle_count, spindle_det = logic_tools.find_class(data, SPINDLE)

            if double_flat_bb_count == 1 and spindle_count == 1:
                double_flat_bb_det = double_flat_bb_det[0]
                spindle_det = spindle_det[0]

            if double_flat_bb_count and spindle_count:
                complete_overlap = logic_tools.complete_overlap(double_flat_bb_det, spindle_det)
                if complete_overlap:
                    gui.update_substep(4)
                    sub_conditions[4] = True

        # SUB 5: right hand totally covering double flat bottom bracket
        if not sub_conditions[5] and sub_conditions[4] == True:
            hand_count, hand_det = logic_tools.find_class(data, HAND)
            double_flat_bb_count, double_flat_bb_det = logic_tools.find_class(data, DOUBLEFLATS_BOTTOM_BRACKET)

            if hand_count > 1:
                _, R_hand = logic_tools.RL_hands(hand_det)

            else:
                R_hand = hand_det[0]

            if double_flat_bb_count == 1:
                double_flat_bb_det = double_flat_bb_det[0]

            if double_flat_bb_count:
                complete_overlap = logic_tools.complete_overlap(R_hand, double_flat_bb_det)
                if complete_overlap:
                    gui.update_substep(5)
                    sub_conditions[5] = True

        # SUB 6: Hand is Out of Field
        if not sub_conditions[6] and sub_conditions[5] == True:
            hand_count, hand_det = logic_tools.find_class(data, HAND)
            if hand_count == 0:
                gui.update_substep(6)
                sub_conditions[6] = True

        # SUB 7: Reconfirm Doubleflatbottombracket is completely over spindle
        if not sub_conditions[7] and sub_conditions[6] == True:
            double_flat_bb_count, double_flat_bb_det = logic_tools.find_class(data, DOUBLEFLATS_BOTTOM_BRACKET)
            spindle_count, spindle_det = logic_tools.find_class(data, SPINDLE)

            if double_flat_bb_count == 1 and spindle_count == 1:
                double_flat_bb_det = double_flat_bb_det[0]
                spindle_det = spindle_det[0]

            if double_flat_bb_count and spindle_count:
                complete_overlap = logic_tools.complete_overlap(double_flat_bb_det, spindle_det)
                if complete_overlap:
                    gui.update_substep(7)
                    sub_conditions[7] = True

        # SUB 8: Only Doubleflatbottombracket and spindle left behind
        if not sub_conditions[8] and sub_conditions[7] == True:
            double_flat_bb_count, double_flat_bb_det = logic_tools.find_class(data, DOUBLEFLATS_BOTTOM_BRACKET)
            spindle_count, spindle_det = logic_tools.find_class(data, SPINDLE)

            if num_class_detected == 2 and spindle_count == 1 and double_flat_bb_count == 1:
                gui.update_substep(8)
                sub_conditions[8] = True

        # if spindle + bolt + hand overlap --> passed

        # if spindle + bolt + hand location aroudn there in the middle = pass

        # could add time duration for them

        if all(sub_conditions):
            print("Step 2 Done")
            gui.mark_step_done(DONE)

def step3_validator():
    # Detections Expected: Left Hand, Right Hand, Double-flats Wrench, (DoubleFlatBottomBracket), (Spindle)
    s4_prev_wrench_xmin = math.inf
    s4_prev_wrench_ymax = math.inf
    s4_prev_wrench_ymin = math.inf
    s4_turn_count = 0
    s4_history = []

    sub_conditions = [False for i in range(6)]
    while current_step == 2:
        data = cv_queue.get()
        num_class_detected = len(data)

        # find double flat wrench
        if not sub_conditions[0]:
            double_flat_wrench_count, _ = logic_tools.find_class(data, DOUBLE_FLATS_WRENCH)
            if double_flat_wrench_count == 1:
                gui.update_substep(0)
                sub_conditions[0] = True

        # find hands
        if not sub_conditions[1] and sub_conditions[0] == True:
            hand_count, hands_det = logic_tools.find_hands(data)
            if hand_count > 1:
                gui.update_substep(1)
                sub_conditions[1] = True

        # overlap hands and doubleflatwrench
        if not sub_conditions[2] and sub_conditions[1] == True:
            hand_count, hand_det = logic_tools.find_class(data, HAND)
            wrench_count, wrench_det = logic_tools.find_class(data, DOUBLE_FLATS_WRENCH)

            for hand in hand_det:
                overlapping, _ = logic_tools.is_overlapping(hand, wrench_det[0])
                if overlapping:
                    gui.update_substep(2)
                    sub_conditions[2] = True

        # complete overlap of wrench over doubleflatbracket
        if not sub_conditions[3] and sub_conditions[2] == True:
            double_flat_bb_count, double_flat_bb_det = logic_tools.find_class(data, DOUBLEFLATS_BOTTOM_BRACKET)
            wrench_count, wrench_det = logic_tools.find_class(data, DOUBLE_FLATS_WRENCH)

            if double_flat_bb_count == 1 and wrench_count == 1:
                double_flat_bb_det = double_flat_bb_det[0]
                wrench_det = wrench_det[0]

            if double_flat_bb_count and wrench_count:
                complete_overlap = logic_tools.complete_overlap(wrench_det, double_flat_bb_det)
                if complete_overlap:
                    gui.update_substep(3)
                    sub_conditions[3] = True

                    # rotation detected and wrench completely over doubleflatbracket
        if not sub_conditions[4] and sub_conditions[3] == True:
            double_flat_bb_count, double_flat_bb_det = logic_tools.find_class(data, DOUBLEFLATS_BOTTOM_BRACKET)
            wrench_count, wrench_det = logic_tools.find_class(data, DOUBLE_FLATS_WRENCH)

            if double_flat_bb_count == 1 and wrench_count == 1:
                double_flat_bb_det = double_flat_bb_det[0]
                wrench_det = wrench_det[0]

            if double_flat_bb_count and wrench_count:
                complete_overlap = logic_tools.complete_overlap(wrench_det, double_flat_bb_det)

                if complete_overlap:
                    # Note: in yolov7 higher y value means lower position in canvas
                    if ((wrench_det[0] < s4_prev_wrench_xmin + 50 and wrench_det[3] < s4_prev_wrench_ymax + 20) or
                            (wrench_det[0] > s4_prev_wrench_xmin + 50 and wrench_det[
                                1] < s4_prev_wrench_ymin + 20)):
                        s4_history.append("Turning")

                    if ((wrench_det[0] > s4_prev_wrench_xmin + 50 and wrench_det[3] > s4_prev_wrench_ymax + 20) or
                            (wrench_det[0] < s4_prev_wrench_xmin + 50 and wrench_det[
                                1] > s4_prev_wrench_ymin + 20)):
                        s4_history.append("Resetting")

                        temp_dict = dict(Counter(s4_history[-10:]))
                        if (temp_dict["Resetting"] > 3):
                            s4_history = []
                            s4_turn_count += 1
                            print(f"Finished Turn: {s4_turn_count}")

                    # use for turn calibrating
                    a = dict(Counter(s4_history))
                    # print(a)

                    s4_prev_wrench_xmin = wrench_det[0]
                    s4_prev_wrench_ymin = wrench_det[1]
                    s4_prev_wrench_ymax = wrench_det[3]

            # Final turn didn't count
            if wrench_count == 0 and a['Turning'] > 0:
                s4_turn_count += 1
                print(f"Finished Turn: {s4_turn_count}")

            if s4_turn_count == 3:
                gui.update_substep(4)
                sub_conditions[4] = True

        # hand is out of field
        if not sub_conditions[5] and sub_conditions[4] == True:
            hand_count, hand_det = logic_tools.find_class(data, HAND)
            if hand_count == 0:
                gui.update_substep(5)
                sub_conditions[5] = True

        # correct overlap increase/decrease

        # correct wrench location or hand location

        # time duration

        if all(sub_conditions):
            print("Step 3 Done")
            gui.mark_step_done(DONE)
            
def step4_validator():
    # Detections Expected: Left Hand, Right Hand, CrankArm, (DoubleFlatBottomBracket), (Spindle)
    sub_conditions = [False for i in range(4)]
    while current_step == 3:
        data = cv_queue.get()
        num_class_detected = len(data)

        # find crank arm
        if not sub_conditions[0]:
            double_flat_wrench_count, _ = logic_tools.find_class(data, CRANK_ARM)
            if double_flat_wrench_count == 1:
                gui.update_substep(0)
                sub_conditions[0] = True

        # find hand
        if not sub_conditions[1] and sub_conditions[0] == True:
            hand_count, hands_det = logic_tools.find_hands(data)
            if hand_count > 1:
                gui.update_substep(1)
                sub_conditions[1] = True

        # find correct overlap
        if not sub_conditions[2] and sub_conditions[1] == True:
            over_count, over_det, g = logic_tools.find_overlapping(data)
            if over_count == 1:
                single_overlap_pair = over_det[0]
                # if the overlapping is between spindle and hand
                if ((single_overlap_pair[0][5] == CRANK_ARM and single_overlap_pair[1][5] ==
                     HAND) or
                        (single_overlap_pair[0][5] == HAND and single_overlap_pair[1][5] ==
                         CRANK_ARM)):
                    gui.update_substep(2)
                    sub_conditions[2] = True

        # correct location
        # hand out of field
        if not sub_conditions[3] and sub_conditions[2] == True:
            hand_count, hand_det = logic_tools.find_class(data, HAND)
            if hand_count == 0:
                gui.update_substep(3)
                sub_conditions[3] = True

        # time duration

        # correct increase/decrease

        if all(sub_conditions):
            print("Step 4 Done")
            gui.mark_step_done(DONE)
        
def step5_validator():
    # Detections Expected: Left Hand, Right Hand, Bolt, CrankArm
    sub_conditions = [False for _ in range(5)]
    bolt_time = 0
    away_time = 0
    while current_step == 4:
        data = cv_queue.get()
        num_class_detected = len(data)

        # SUB 0 : is there a hand?
        if not sub_conditions[0]:
            hand_count, hands_det = logic_tools.find_hands(data)
            if hand_count:
                # procedure[current_step].update_description(emoji.emojize("Found Hands 👍"))
                gui.update_substep(0)
                sub_conditions[0] = True
        # ['found hands', 'found crank arm', 'screwing bolt into crank arm', 'screwed bolt into crank arm'']
        # SUB 1 : is there a pedal wrench? (index = 7)
        if not sub_conditions[1]:
            crank_count, crank_det = logic_tools.find_class(data, 12)
            if crank_count:
                # procedure[current_step].update_description(u'Found crank arm👍')
                gui.update_substep(1)
                sub_conditions[1] = True
        # Add rotating condition
        # SUB 2 : are they overlapped? hand holding spindle?
        over_count = 0
        over_dict = {}
        if num_class_detected > 1:
            over_count, over_det, over_dict = logic_tools.find_overlapping(data)
        # print(over_dict)
        # SUB 3 : leaving right hand + increasing left hand
        if not sub_conditions[3] and sub_conditions[1] == True:
            hand_i_bolt = over_dict.get((4, 10)) or over_dict.get((10, 4))
            crank_i_bolt = over_dict.get((12, 10)) or over_dict.get((10, 12))
            if crank_i_bolt and hand_i_bolt:
                if hand_i_bolt < 0.015 and crank_i_bolt < 0.015:
                    bolt_time += 1
                    if not sub_conditions[2]:
                        # procedure[current_step].update_description(u'Screwing bolt into crank arm...')
                        gui.update_substep(2)
                        sub_conditions[2] = True
                    if bolt_time > 30:
                        # procedure[current_step].update_description(u'Screwed bolt into crank arm')
                        gui.update_substep(3)
                        sub_conditions[3] = True
        elif not sub_conditions[4] and sub_conditions[3]:
            hand_i_bolt = over_dict.get((4, 10)) or over_dict.get((10, 4))
            if not hand_i_bolt:
                away_time += 1
                if (away_time > 8) and not (over_dict.get((4, 12)) or over_dict.get((12, 4))):
                    if not (over_dict.get((4, 12)) or over_dict.get((12, 4))):
                        gui.update_substep(4)
                        sub_conditions[4] = True

        if all(sub_conditions[0:5]):
            print("Step 5 done")
            gui.mark_step_done(DONE)
            
def step6_validator():
    pedal_time = 0
    sub_conditions = [False for _ in range(7)]
    while current_step == 5:
        # ['found hands', 'found pedal', 'hand holding pedal', 'screwing pedal into crank', 'screwed pedal into crank']
        data = cv_queue.get()
        num_class_detected = len(data)

        # SUB 0 : is there a hand?
        if not sub_conditions[0]:
            hand_count, hands_det = logic_tools.find_hands(data)
            if hand_count:
                # procedure[current_step].update_description(emoji.emojize("Found Hands 👍"))
                gui.update_substep(0)
                sub_conditions[0] = True

        # SUB 1 : is there a pedal wrench? (index = 7)
        if not sub_conditions[1] and sub_conditions[0] == True:
            ped_count, ped_det = logic_tools.find_class(data, 11)
            if ped_count:
                # procedure[current_step].update_description(u'Found pedal👍')
                gui.update_substep(1)
                sub_conditions[1] = True
        # Add rotating condition
        # SUB 2 : are they overlapped? hand holding spindle?
        over_count = 0
        over_dict = {}
        if num_class_detected > 1:
            over_count, over_det, over_dict = logic_tools.find_overlapping(data)
        # print(over_dict)
        # SUB 3 : leaving right hand + increasing left hand
        if not sub_conditions[2] and sub_conditions[1] == True:
            if over_dict.get((4, 11)) or over_dict.get((11, 4)):
                # procedure[current_step].update_description(u'Hand holding pedal 👍')
                gui.update_substep(2)
                # print("Intersection of hand and pedal: ",over_dict.get((4,11)) or over_dict.get((11,4)))
                sub_conditions[2] = True
        if not sub_conditions[4] and sub_conditions[1] == True:
            crank_i_pedal = over_dict.get((12, 11)) or over_dict.get((12, 11))
            # crank_i_bolt = over_dict.get((12,10)) or over_dict.get((10,12))
            if crank_i_pedal:
                if crank_i_pedal < 0.05:
                    pedal_time += 1
                    if not sub_conditions[3]:
                        if pedal_time > 2:
                            # procedure[current_step].update_description(u'Screwing pedal into crank arm...')
                            gui.update_substep(3)
                            sub_conditions[3] = True
                    elif pedal_time > 45:
                        # procedure[current_step].update_description(u'Screwed pedal into crank arm')
                        gui.update_substep(4)
                        sub_conditions[4] = True
        if sub_conditions[4] and not sub_conditions[5]:
            hand_i_pedal = over_dict.get((4, 11)) or over_dict.get((4, 11))
            if not hand_i_pedal:
                gui.update_substep(5)
                sub_conditions[5] = True
        if all(sub_conditions[0:6]):
            print("everything done")
            gui.mark_step_done(DONE)

def step7_validator():
    """
    StepValidator for step 7: using pedal lockring wrench to install pedal
    objects of interest in view:
        - hand(s)
        - pedal wrench
        - crank arm
        - pedal
        - bolt
    TODO:   1) does logic stop when override/revert button is clicked?
            2) check for null in detection [0]
    """

    # sanity check for current step
    if current_step != 6:
        return -1
    
    # current_step = 6
    # Detections Expected: Left Hand, Right Hand, Pedal Locking wrench, Pedal, CrankArm, Bolt
    
    # a lil timer
    start_time = time.time()
    print(f"Started step 7")

    """
    A) Initial stage condition to be satisfied (i.e. condition for starting stage):
        - pedal is in (proximity) of crank arm: 
            1) pedal intersects with crank arm 
            2) pedal bbox and crank bbox centers are within max + threshold from each other (???)
        - bolt is in crank arm: bolt bbox is (mostly) fully within crank arm bbox (given camera is overhead)
    Q: Do these conditions have to be constantly validated throughout the step?
    """

    initial_stage_satisfied = False
    condition_persistor = Persistor(frames=30, condition_name="Initial Stage")
    while not initial_stage_satisfied:
        t1 = time.time()
        """
        TODO:   don't just take the first reading lol
                'proximity' might not be strict enough? (proximity + edge of crank arm?)
        """
        data = cv_queue.get()  # [[xyxy(4), conf(1), class(1)], ...]
        # print(f"These are not the droids you're looking for.")

        # if necessary objs do not exist (caveat: glitch/hidden for brief moment)
        if len(data[data[:, 5] == PEDAL]) == 0 or len(data[data[:, 5] == CRANK_ARM]) == 0 or len(
                data[data[:, 5] == BOLT]) == 0:
            continue

        #  Taking the first of each class of interest found.
        pedal = data[data[:, 5] == PEDAL][0]
        crank_arm = data[data[:, 5] == CRANK_ARM][0]
        bolt = data[data[:, 5] == BOLT][0]

        # init condition 1: pedal in prox of crank arm;
        pedal_crank_iou = bbox_iou(crank_arm[:4], pedal[:4])
        bolt_crank_iou = bbox_iou(crank_arm[:4], bolt[:4])
        # print(f"crank_arm={crank_arm}, pedal={pedal}, bolt={bolt}")
        # print(f"bolt_crank={bbox_intersection(bolt, crank_arm)}/{bbox_area(bolt)}")
        # print(f"pedal_crank={pedal_crank_iou}\n")

        if pedal_crank_iou < 0.05 or pedal_crank_iou > 0.2 or bbox_intersection(bolt, crank_arm) + 0.001 < bbox_area(
                bolt):
            condition_persistor.reset()
            continue

        if condition_persistor.verify():
            initial_stage_satisfied = True
            # print(f"{time.time() - t1} seconds")
        else:
            condition_persistor.persist()

    gui.update_substep(0)  # substep 1 satisfied

    """
    B) In-progress conditions to be satisfied:
        1. hand intersecting greatly with pedal wrench (for the duration of rotation being sensed/detected)
            - this sub step is active as long as pedal wrench+hand combination is engaged with pedal
            - rotation in bbox can be very roughly sort-of-ish detected using the center of the bbox; 
                but the rotation is not a smooth circle/ellipse
        2. ?
    Concerns:
        - losing visual (use EWMA on 4 point of bbox to approximate expected location until next update? == tracking???)
            - use last location? But what if obj is hidden for a long time? 
                (& differentiate between removed from frame vs hidden??????? this is so extra wtf) 
    """
    in_progress_stage_satisfied = False
    num_rotations = 0
    condition_persistor = Persistor(frames=10, condition_name="1 rotation")
    MIN_ROTATION = 3
    global sensor_queue, sensor_in_use
    while not in_progress_stage_satisfied:
        # print("still running...")
        data = cv_queue.get()  # [[xyxy(4), conf(1), class(1)], ...]

        # if necessary objs do not exist (caveat: glitch/hidden for brief moment)
        if len(data[data[:, 5] == PEDAL_LOCKRING_WRENCH]) == 0 or len(data[data[:, 5] == PEDAL]) == 0 or len(
                data[data[:, 5] == HAND]) == 0:
            continue

        if not sensor_in_use.is_set(): sensor_in_use.set()
        # print("Sensor set")

        pedal = data[data[:, 5] == PEDAL][0]
        hands = data[data[:, 5] == HAND]
        pedal_wrench = data[data[:, 5] == PEDAL_LOCKRING_WRENCH][0]

        sensor_data = sensor_queue.get()
        # print(sensor_data, "\n")
        if sensor_data['rotating']:
            # print(f"detecting rotation...({sensor_data['num_rotations']}/3)")
            pedal_pedal_lockring_iou = bbox_iou(pedal_wrench[:4], pedal[:4])
            # takes max to determine the iou of the most likely hand holding wrench (???)
            hand_pedal_lockring_iou = max([bbox_iou(hand[:4], pedal_wrench[:4]) for hand in hands])

            if sensor_data['num_rotations'] != num_rotations:
                gui.update_substep(sensor_data['num_rotations'])
                num_rotations = sensor_data['num_rotations']

            if num_rotations < MIN_ROTATION and condition_persistor.verify():
                # one rotation completed. Reset Persistor for next rotation
                condition_persistor.reset()

            if pedal_pedal_lockring_iou > 0.1 and hand_pedal_lockring_iou > 0:
                condition_persistor.persist()
                # TODO: when to reset on error?
            else:
                continue

            # print(f"num rotations = {num_rotations}")
            if sensor_data['num_rotations'] >= MIN_ROTATION and condition_persistor.verify():
                in_progress_stage_satisfied = True

    sensor_in_use.clear()
    
    gui.mark_step_done(DONE)
    terminate.set()
    
    return time.time() - start_time


# ===================== GUI CLASS ==============================


class DisplayGUI:
    def __init__(self, app):
        """
        Initializations of the GUI components for the loading screen, starts loading thread
        :param app: a TK root object
        """
        self.app = app
        self.app.title("Project Pete")

        # "responsive" sizing
        self.min_width = int(self.app.winfo_screenwidth() * 0.85)
        self.min_height = int(self.app.winfo_screenheight() * 0.7)
        self.app.minsize(width=self.min_width, height=self.min_height)

        # Ensure closing of detect thread on quit
        self.app.protocol('WM_DELETE_WINDOW', self.close_app)

        self.loading_frame = tk.Frame(self.app, bg=space_grey_background)
        self.loading_frame.pack(fill="both", expand=True)

        logo = ImageTk.PhotoImage(Image.open('pete.png').resize((445, 200)))
        self.logo_label = tk.Label(self.loading_frame, bg=space_grey_background)
        self.logo_label.pack(pady=50)
        self.logo_label.config(image=logo)
        self.logo_label.image = logo

        # Loading gif
        self.loading_wheel_label = tk.Label(self.loading_frame, bg=space_grey_background)
        self.loading_wheel_label.pack(pady=50)

        loading_thread = threading.Thread(target=self._update_loading_gif, args=[])
        loading_thread.daemon = True
        loading_thread.start()

        self.loading_label = tk.Label(self.loading_frame, fg='white', text="Please wait, loading model",
                                      bg=space_grey_background,
                                      font=("Arial", 24, 'bold'))
        self.loading_label.pack(pady=25)

        loading_model_thread = threading.Thread(target=self._check_loaded_model, args=[])
        loading_model_thread.daemon = True
        loading_model_thread.start()

    def close_app(self):
        self.app.destroy()

    def procedure_tracking_setup(self, app):
        """
        Initializations of the GUI components for the actual procedure tracking system.
        :param app: a TK root object
        """
        global procedure, current_step

        # Video Frame ========================================================
        # Create a frame for the live stream
        self.left_frame = tk.Frame(self.app, bg=dark_theme_background, width=0.7 * self.min_width)
        self.left_frame.pack(side="left", fill="both", expand=False)

        # Create a label to display stream
        self.livestream = tk.Label(self.left_frame, )
        self.livestream.pack(padx=(80, 50), pady=(40, 0))

        # Logs ========================================================
        lw = int(self.left_frame.winfo_screenwidth() * 0.5)

        # Performance
        self.performance = tk.Frame(self.left_frame, bg=dark_theme_background, width=lw)
        self.performance.pack(side="left", fill="both", expand=False)
        self.performance_header = tk.Label(self.performance, text="Performance", bg=dark_theme_background,
                                           font=("Arial", 24, 'bold'))
        self.performance_header.pack(padx=(80, 0), pady=(10, 0))

        # Create a label to display the runtime
        self.runtime_label = tk.Label(self.performance, bg=dark_theme_background, text="Runtime: 0 seconds")
        self.runtime_label.pack(padx=(100, 0))

        # Create a thread for updating the runtime label
        runtime_thread = threading.Thread(target=self._update_runtime)
        runtime_thread.daemon = True
        runtime_thread.start()

        # Substep Progress
        self.substep = tk.Frame(self.left_frame, width=lw, bg=dark_theme_background)
        self.substep.pack(padx=(80,0), pady=(10,0), side="left", fill="both", expand=True)
        self.substep_header = tk.Label(self.substep, text="Substep Progress", font=("Arial", 24, 'bold'), justify="center", bg=dark_theme_background)
        self.substep_header.pack(fill='x')

        # list of Tkinter labels for substeps
        self.substep_list = []

        # Procedure List ===================================================
        # Create a frame for the steps list (30% of total width)
        self.right_frame = tk.Frame(app, width=0.3 * self.min_width, bg=dark_theme_background)
        self.right_frame.pack(side="right", fill="both", expand=True)

        # Scrollable list view
        self.canvas = tk.Canvas(self.right_frame, bg=dark_theme_background, borderwidth=0, highlightthickness=0)
        self.procedure_list = tk.Frame(self.canvas, bg=dark_theme_background, borderwidth=0, highlightthickness=0)
        self.scrollbar = ScrollBar(self.procedure_list, orient="vertical", command=self.canvas.yview)
        self.procedure_list.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="top", fill="both", expand=True, pady=(40, 0))
        self.scrollbar.pack(side="left", fill="y", expand=False)
        self.canvas.create_window((0, 0), window=self.procedure_list, anchor="nw", tags="canvas_frame")

        procedure = self.get_procedure()
        # initialize lists in GUI
        for step in procedure:
            if step.status == IN_PROGRESS: current_step = step.index
            step.build(self.procedure_list)

        self.procedure_list.pack(side="right", fill="both",
                                 expand=True)  # pack after resizing ensures procedure list is correct size

        # Button frame
        self.button_frame = tk.Label(self.right_frame, fg='white', bg=dark_theme_background, borderwidth=5)
        self.button_frame.pack(fill="x", expand=False)
        
        # Revert button
        self.revert = tk.Label(self.button_frame, fg='white', bg=revert_button_color, text="Revert - undo step",
                               borderwidth=5)

        # Override button
        self.override = tk.Label(self.button_frame, fg='white', bg=override_button_color, text="Override - mark done",
                                 borderwidth=5)

        self.revert.pack(fill='both',expand=True, padx=(20, 10), pady=(10, 10),side='left')
        self.revert.bind("<ButtonRelease-1>", self.revert_mark_done)
        self.override.pack(fill='both',expand=True, padx=(10, 20), pady=(10, 10), side='right')
        self.override.bind("<ButtonRelease-1>", self.override_mark_done)

        # Decision logic ===================================
        # Separate thread for communicating with other functions to get decision result
        logic_thread = threading.Thread(target=decision_logic)
        logic_thread.daemon = True
        logic_thread.start()

    def get_procedure(self):
        """
        Gets steps in a procedure.
        Step is a defined class. More stuff can be added in there to help with validation.
        TODO: actual steps lol
        """
        global procedure, current_step

        # clear out and initialize procedure + step count
        procedure = []
        current_step = 0

        # dummy steps
        # TODO: define steps & their individual criteria

        for i in range(0, 7):
            if i == 0:
                title = f"Step {i + 1}, Spindle Installation"
                description = "Put the spindle (rod-like object in left image) into the axle hole \
                    \nOnce complete, it should look like the image on the right"
                status = NOT_DONE
                substeps = ['1.1 - Detect Hands',
                            '1.2 - Detect Spindle',
                            '1.3 - Hand holding Spindle',
                            '1.4 - Passed to Left Hand',
                            '1.5 - Spindle on Left Hand',
                            '1.6 - Spindle leaves Left Hand',
                            '1.7 - Spindle Alone']
                pictures = ['step1.1.png', 'step1.2.png']

            if i == 1:
                title = f"Step {i + 1}, Bottom Bracket Installation and Tightening"
                description = "Place the Double Flat Bottom Bracket into the axle hole \
                    \nThen, turn it clockwise with you fingers to tighten it"
                status = NOT_DONE
                substeps = ['2.1 - Detect Hands',
                            '2.2 - Detect Fouble Flat Bottom Bracket',
                            '2.3 - Detect Spindle',
                            '2.4 - Single Hand Holding Double Flat Bottom Bracket',
                            '2.5 - Detect Complete Overlap of Double Flat Bottom Bracket over Spindle',
                            '2.6 - Detect Right Hand completely over Double Flat Bottom Bracket',
                            '2.7 - Hand Out of Field',
                            '2.8 - Confirm Double Flat Bottom Bracket completely overlaps Spindle',
                            '2.9 - Double Flat Bottom Bracket and Spindle left behind only']
                pictures = ['step2.png']

            if i == 2:
                title = f"Step {i + 1}, Tighten with Double Flat Wrench"
                description = "Use the Double Flat Wrench to tighten the Double Flat Bottom Bracket by turning it clockwise"
                status = NOT_DONE
                substeps = ['3.1 - Detect Double Flat Wrench',
                            '3.2 - Detect Hands',
                            '3.3 - Detect Hand Overlap',
                            '3.4 - Complete Overlap of Wrench over Double Flat Bottom Bracket',
                            '3.5 - Tighten by THREE Rotations and Complete overlap Detected',
                            '3.6 - Hand Out of Field']
                pictures = ['step3.png']

            if i == 3:
                title = f"Step {i + 1}, Crank Arm Installation"
                description = "Place the Crank Arm into the axle hole"
                status = NOT_DONE
                substeps = ['4.1 - Detect Crank Arm',
                            '4.2 - Detect Hands',
                            '4.3 - Detect Single Hand Overlap over Crank Arm',
                            '4.4 - Hand Out of Field']
                pictures = ['step4.png']

            if i == 4:
                title = f"Step {i + 1}, Bolt Installation"
                description = "Secure the Crank Arm with the little bolt (bolt in left image) by placing it into the axle hole\
                    \n Then, turn it clockwise with your fingers to tighten it"
                status = NOT_DONE
                substeps = ['5.1 - Detect Hands',
                            '5.2 - Detect Crank Arm',
                            '5.3 - Screwing Bolt into Crank Arm',
                            '5.4 - Screwed Bolt into Crank Arm',
                            '5.4 - Detached Hands and Bolt']
                pictures = ['step5.1.png', 'step5.2.png']

            if i == 5:
                title = f"Step {i + 1}, Pedal Installation"
                description = "Place the pedal into the other side of the Crank Arm \
                \nThen, tighten the bolt on the other side of the pedal to secure it"
                status = NOT_DONE
                substeps = ['6.1 - Detect Hands',
                            '6.2 - Detect Pedal',
                            '6.3 - Hand holding Pedal',
                            '6.4 - Screwing Pedal into Crank',
                            '6.5 - Screwed Pedal into Crank',
                            '6.6 - Detached Hand and Pedal']
                pictures = ['step6.png']

            if i == 6:
                title = f"Step {i + 1}, Pedal Tightening with Crank Arm"
                description = "Use the Pedal Locking Wrench (left image) to secure the bolt on the other side of the pedal"
                status = NOT_DONE
                substeps = ['7.1 - Pedal is placed in Bolted-down Crank Arm',
                            '7.2 - Secure Pedal using Pedal Wrench (rotation 1/3)',
                            '7.3 - Secure Pedal using Pedal Wrench (rotation 2/3)',
                            '7.4 - Secure Pedal using Pedal Wrench (rotation 3/3)']
                pictures = ['step7.1.png', 'step7.2.png']

            s = Step(i, title, description, status, substeps, pictures)

            procedure.append(s)

        return procedure

    def mark_step_done(self, done_type):
        """
        Mark step as done and go to next step (if exists)
        :param done_type: DONE_OV (overriden), DONE (regular auto-approved)
        current_step is 0-indexed
        procedure is 0-indexed
        """
        global current_step, procedure

        isLastStep = current_step == len(procedure) - 1
        procedure[current_step].update_status(done_type, isFocus=isLastStep)

        self.canvas.yview_moveto(1.0)
        if isLastStep: return

        self.clear_substeps()
        current_step += 1
        procedure[current_step].update_status(IN_PROGRESS, isFocus=True)
        self.build_substeps(procedure[current_step])

    def override_mark_done(self, e):
        """
        Overrides logic decision (mark as complete - OV)
        """
        self.mark_step_done(DONE_OV)

    def revert_mark_done(self, e):
        global current_step, procedure

        if current_step == 0: return  # check if first step

        # allow for reverting last step
        self.clear_substeps()
        isLastStep = current_step == len(procedure) - 1
        if isLastStep and (procedure[current_step].status == DONE or procedure[current_step].status == DONE_OV):
            procedure[current_step].update_status(IN_PROGRESS)
        else:
            current_step -= 1
            procedure[current_step].update_status(IN_PROGRESS)
            procedure[current_step + 1].update_status(NOT_DONE, isFocus=False)
        self.build_substeps(procedure[current_step])
        self.canvas.yview_moveto(-1.0)

    def set_frame(self, frame):
        """
        Updates detection preview on the left
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (750, 450))
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.livestream.config(image=photo)
        self.livestream.image = photo

    def build_substeps(self, step):
        for i, s in enumerate(step.substeps):
            temp = tk.Label(self.substep, text=s,justify='left',anchor='w', bg=dark_theme_background, font=("Arial", 16))
            temp.pack(fill='x')
            self.substep_list.append(temp)

    def update_substep(self, index):
        if self.substep_list[index]['text'][-1] == "\u2713":
            return
        self.substep_list[index]['fg'] = substep_complete_text_color
        self.substep_list[index]['text'] += " \u2713 "

    def clear_substeps(self):
        for _ in range(len(self.substep_list)):
            temp = self.substep_list.pop()
            temp.destroy()

    def _check_loaded_model(self):
        """
        Checks if model has finished loading, when finished, it will initialize the procedure tracking GUI and start detection
        """
        detect_thread = threading.Thread(target=detect, args=[])
        detect_thread.daemon = True
        detect_thread.start()

        detect_ready.wait() # let GUI start up

        self.loading_frame.destroy()

        self.procedure_tracking_setup(self.app)

        # Sensor  ===================================
        sensor_thread = threading.Thread(target=sensor_detect, args=[])
        sensor_thread.daemon = True
        sensor_thread.start()

    def _update_loading_gif(self):

        frames = Image.open('loading.gif').n_frames
        loading_frames = []

        # load frames
        for i in range(frames):
            temp = tk.PhotoImage(file='loading.gif', format=f"gif -index {i}")  # TODO: downscale loading = less crunchy
            loading_frames.append(temp)

        i = 0
                
        while not detect_ready.is_set():
            i = i + 1
            i = i % frames

            self.loading_wheel_label.config(image=loading_frames[i])
            self.loading_wheel_label.image = loading_frames[i]
            time.sleep(0.1)

    def _update_runtime(self):
        """
        Helper function to update runtime clock
        """
        start_time = time.time()
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Format elapsed time into hh:mm:ss
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)

            runtime_str = f"Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}"
            self.runtime_label.config(text=runtime_str)

            time.sleep(1)  # Update the label every 1 second


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='Demo_Only_B40.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='test_videos/bottomBracketInstall.MOV',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default='True', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', default='True', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', default='True', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    root = tk.Tk()
    gui = DisplayGUI(root)

    root.mainloop()

    exit()  # close program and all other threads after destroy
