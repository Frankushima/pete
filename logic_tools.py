#**************************************
#
# Helper Functions for Decision Logic
#
#**************************************

# print(names[int(det[0][5])]) in GUI_detect.py gives the following
# ['adjustable monkey wrench', 
# 'monkey wrench', 
# 'allen key', 
# 'double-flats wrench', 
# 'hand', 
# 'pedal lockring wrench', 
# 'crank remover', 
# 'spindle', 
# 'doubleFlatsBottomBracket', 
# 'crankArmNonChainSide', 
# 'bolt', 
# 'pedal', 
# 'crankArm']

from utils.general import bbox_iou

from enum import Enum

import math

class_index = {
    'adjustablemonkeywrench': 0,
    'monkeywrench': 1,
    'allenkey': 2,
    'doubleflatswrench': 3,
    'hand': 4,
    'pedallockringwrench': 5,
    'crankremover': 6,
    'spindle': 7,
    'doubleFlatsBottomBracket': 8,
    'crankArmNonChainSide': 9,
    'bolt': 10,
    'pedal': 11,
    'crankArm': 12
}

class Trendline(Enum):
    INITIALIZE = -1
    DECREASING = 0
    INCREASING = 1

def get_box_center(x1, y1, x2, y2):
    return (x1+x2)/2, (y1+y2)/2

# point1 = (x1, y1)
# point2 = (x2, y2)
def get_euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# box_coordinates in format [x1, y1, x2, y2]
def are_they_the_same_detections(box_a_det, box_b_det, threshold=0):
    center_a = get_box_center(*box_a_det[:4])
    center_b = get_box_center(*box_b_det[:4])

    distance = get_euclidean_distance(center_a, center_b)

    if box_a_det[5] != box_b_det[5]:
        return False, distance

    if distance > threshold:
        return False, distance
    
    return True, distance

def find_hands(det):
    if not len(det):
        return
    
    count = 0
    hands = []

    for each_det in det:
        # index of 4 is currently hand
        if each_det[5] == class_index['hand']:
            count += 1
            hands.append(each_det)

    return count, hands

def RL_hands(hands_det):        
    if len(hands_det) != 2:
        print("Not enough hands")
        return
    
    hand1 = hands_det[0]
    hand2 = hands_det[1]

    center_hand1 = get_box_center(*hand1[:4])
    center_hand2 = get_box_center(*hand2[:4])

    if(center_hand1[0] < center_hand2[0]):
        L_hand = hand1
        R_hand = hand2
    else:
        L_hand = hand2
        R_hand = hand1

    return L_hand, R_hand

def find_class(det, cls_id):
    if not len(det):
        return
    
    count = 0
    target_class = []

    for each_det in det:
        if each_det[5] == cls_id:
            count += 1
            target_class.append(each_det)

    return count, target_class

def find_overlapping(det):
    overlapping_detections = []

    for a in range(len(det)):
        for b in range(a + 1, len(det)):
            iou = bbox_iou(det[a][:4], det[b][:4])
            if iou > 0:  
                overlapping_detections.append((det[a], det[b], iou))

    overlapping_pairs_count = len(overlapping_detections)

    return overlapping_pairs_count, overlapping_detections

def is_overlapping(detA, detB):
    iou = bbox_iou(detA[:4], detB[:4])
    if iou > 0:
        return True, iou
    else:
        return False, iou
