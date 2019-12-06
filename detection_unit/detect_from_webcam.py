# Import dependencies

import time
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import torchvision

import argparse

from image_detector import detect_image

def detect_from_webcam(model_file=None, threshold=0.5):
    vs = cv2.VideoCapture(0)

    while True:
    
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out_image = detect_image(image=Image.fromarray(frame), num_classes=4,model_file=model_file, show_image=False, threshold=threshold)
           
        cv2.imshow('frame',np.asarray(out_image[0]))



    print("[INFO] cleaning up...")
    vs.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    
    # initiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="path to model file")
    parser.add_argument("-t", "--threshold", required=True, help="confidence threshold value")

    
    args = vars(parser.parse_args())

    detect_from_webcam(model_file=args["model"], threshold=float(args["threshold"]))




