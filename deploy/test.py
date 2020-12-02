import face_model
import argparse
import cv2
import sys
import numpy as np
import time


parser = argparse.ArgumentParser(description="face model test")
# general
parser.add_argument("--image-size", default="112,112", help="")
parser.add_argument("--model", default="", help="path to load model.")
parser.add_argument("--gpu", default=0, type=int, help="gpu id")
args = parser.parse_args()

model = face_model.FaceModel(
    -1, "../models/model-r50-am-lfw/model", 0, use_large_detector=True
)
for i in range(15):
    t1 = time.perf_counter()
    img = cv2.imread("Tom_Hanks_54745.png")
    img = model.get_input(img)
    cv2.imwrite("img.jpg", img)
    f1 = model.get_feature(img)
    f2 = model.get_feature(img)
    sim = np.dot(f1, f2)
    assert sim >= 0.99 and sim < 1.01
    t2 = time.perf_counter()
    print(f"Time to process: {t2-t1}s.")
