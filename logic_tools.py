#**************************************
#
# Helper Functions for Decision Logic
#
#**************************************
import numpy as np
import torch

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

# in yolov7 higher y value means lower position in canvas
def overlap_by_reference(detA, detB):
    # reference is detB
    detA_xmin, detA_ymin, detA_xmax, detA_ymax = detA[:4]
    detB_xmin, detB_ymin, detB_xmax, detB_ymax = detB[:4]

    # confirm overlap
    overlapping, iou =  is_overlapping(detA, detB)
    if not overlapping:
        return

    # get detB area
    detB_width = detB_xmax - detB_xmin
    detB_height = detA_ymax - detA_ymin

    detB_area = detB_width*detB_height

    # get overlapping area
    xA = max(detA_xmin, detB_xmin)
    yA = max(detA_ymin, detB_ymin)
    xB = min(detA_xmax, detB_xmax)
    yB = min(detA_ymax, detB_ymax)

    overlap_area = (xB-xA) * (yB-yA)

    print(f'Overlap_area = {overlap_area}, detB_area = {detB_area}')

    return overlap_area/detB_area


# complete overlap is seeing if detA completely cover detB
def complete_overlap(detA, detB):
    # reference is detB
    detA_xmin, detA_ymin, detA_xmax, detA_ymax = detA[:4]
    detB_xmin, detB_ymin, detB_xmax, detB_ymax = detB[:4]

    # confirm overlap
    overlapping, iou =  is_overlapping(detA, detB)
    if not overlapping:
        return

    if (detB_xmin >= detA_xmin and
        detB_ymin >= detA_ymin and
        detB_xmax <= detA_xmax and
        detB_ymax <= detA_ymax):
        return True

    else:
        return False

def rotation_detect():
    return










def bbox_intersection(box1, box2):
    # xyxy
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    return inter

def bbox_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def process(file):
    with open(file, 'r') as file:
        lines = [line.strip() for line in file if line.strip() != '']
    lines = [lines[i:i + 4] for i in range(0, len(lines), 4)]

    full = []
    for t, line in enumerate(lines):
        data_dict = {}
        for sensor in line:
            readings = sensor.split()

            key = readings[0]
            if key == "Temp:":
                key = "Temp"
                values = readings[1]
            else:
                values = [float(val) for val in readings[1:] if not ":" in val]
            data_dict[key] = values
        full.append(data_dict)
    return [entry['Angle'][2] for entry in full]

def camera_sensor_frame_match(x, sr=10, fps=17):
    """
    check if the x-th sensor sample matches to a frame in the camera
    param: sr = sampling rate (sensor)
    param: fps = frame rate (cam)
    """
    for i in range(x):
        if i * fps // sr == x: return True
    return False

# ==================== PERSISTOR CLASS =========================
class Persistor:
    def __init__(self, frames: int, condition_name):
        self.condition = frames
        self.counter = 0
        self.name = condition_name

    def persist(self):
        self.counter += 1
        print(f"[{self.name}] Persisted ({self.counter}/{self.condition}).")

    def verify(self):
        return self.counter >= self.condition

    def reset(self):
        if np.random.random() < 0.8:
            self.counter = 0
            print(f"[{self.name}] Disrupted ({self.counter}/{self.condition}).")
