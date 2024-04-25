import time
import GUI_detect
import numpy as np
from shared import *
from utils.general import bbox_iou


def step7_validator(step_GUI):
    """
    StepValidator for step 7: using pedal lockring wrench to install pedal
    objects of interest in view:
        - hand(s)
        - pedal wrench
        - crank arm
        - pedal
        - bolt
    """

    # a lil timer
    start_time = time.time()

    """
    A) Initial stage condition to be satisfied (i.e. condition for starting stage):
        - pedal is in (proximity) of crank arm: 
            1) pedal intersects with crank arm 
            2) pedal bbox and crank bbox centers are within max + threshold from each other (???)
        - bolt is in crank arm: bolt bbox is (mostly) fully within crank arm bbox (given camera is overhead)
    Q: Do these conditions have to be constantly validated throughout the step?
    """

    initial_stage_satisfied = False
    while not initial_stage_satisfied:
        data = np.array(GUI_detect.get_data())  # [[xyxy(4), conf(1), class(1)], ...]

        #  Taking the first of each class of interest found.
        #  TODO: don't just take the first lol
        pedal = data[data[:, 5] == PEDAL][0]
        crank_arm = data[data[:, 5] == CRANK_ARM][0]
        bolt = data[data[:, 5] == BOLT][0]

        # init condition 1: pedal in prox of crank arm;
        pedal_crank_iou = bbox_iou(crank_arm[:4], pedal[:4])
        if pedal_crank_iou < 0.1: continue

        # init condition 2: bolt in crank arm
        bolt_crank_iou = bbox_iou(crank_arm[:4], bolt[:4])
        if pedal_crank_iou < 0.8: continue

        initial_stage_satisfied = True

    # adding some info to step display.
    # TODO: make this prettier or something idk
    step_GUI.update_description("Initial condition satisfied.")

    """
    B) In-progress conditions to be satisfied:
        1. hand intersecting greatly with pedal wrench (for the duration of rotation being sensed/detected)
            - as long as pedal wrench+hand combination is engaged with pedal, this sub step is active 
            - rotation in bbox can be very roughly sort-of-ish detected using the center of the bbox; 
                but the rotation is not a smooth circle/ellipse
    Concerns:
        - losing visual (use EWMA on 4 point of bbox to approximate expected location until next update?)
            - use last location? But what if obj is hidden for a long time? 
                (& differentiate between removed from frame vs hidden) 
    """
    in_progress_stage_satisfied = False
    while not in_progress_stage_satisfied:
        data = np.array(GUI_detect.get_data())  # [[xyxy(4), conf(1), class(1)], ...]
        # (?) needs some sort of persistent tracking here to detect rotation over time

    """
    C) End-stage conditions:
        - pedal is in (proximity) of crank arm
        - bolt is in crank arm: bolt bbox is (mostly) fully within crank arm bbox (given camera is overhead)
        - sensor requirements??? idrk. How do we determine whether the pedal is fully screwed in? TBF even with ground
            assistance, that's not something anyone will know besides the user, unless we get a torque sensor.
    """
    end_stage_satisfied = False
    while not end_stage_satisfied:
        data = np.array(GUI_detect.get_data())  # [[xyxy(4), conf(1), class(1)], ...]

        #  Taking the first of each class of interest found.
        #  TODO: (like initial) don't just take the first
        pedal = data[data[:, 5] == PEDAL][0]
        crank_arm = data[data[:, 5] == CRANK_ARM][0]
        bolt = data[data[:, 5] == BOLT][0]

        # init condition 1: pedal in prox of crank arm;
        pedal_crank_iou = bbox_iou(crank_arm[:4], pedal[:4])
        if pedal_crank_iou < 0.1: continue

        # init condition 2: bolt in crank arm
        bolt_crank_iou = bbox_iou(crank_arm[:4], bolt[:4])
        if bolt_crank_iou < 0.8: continue

        end_stage_satisfied = True

    return start_time - time.time()

