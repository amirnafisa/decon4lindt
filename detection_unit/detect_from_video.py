# Import dependencies

import time
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import torchvision

import argparse

from image_detector import detect_image

def detect_video(video=None, videoPath=None, outputVideoPath=None, modelFile=None, threshold=0.5):
    if videoPath is not None:
        vs = cv2.VideoCapture(videoPath)

    writer = None
    try:
        prop = cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))
    except:
        print("[INFO] could not determine # of frames in video")
        total = -1
    
    i = 0
    
    while True:
    
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        if i%30 == 0:
            start = time.time()
            out_image = detect_image(image=Image.fromarray(frame), num_classes=4,model_file=modelFile, show_image=False, threshold=threshold)[0]
            end = time.time()
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(outputVideoPath, fourcc, 1, (frame.shape[1], frame.shape[0]), True)

                if total > 0:
                    elap = (end - start)
                    print("[INFO] single frame took {:.4f} seconds".format(elap))
                    print("[INFO] estimated total time to finish: ",elap*total/30)
            writer.write(np.asarray(out_image))
        i+=1

    print("[INFO] cleaning up...")
    writer.release()
    vs.release()




if __name__ == "__main__":
    
    # initiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="path to input video file")
    parser.add_argument("-o", "--output", required=True, help="path to output video file")
    parser.add_argument("-m", "--model", required=True, help="path to model file")
    parser.add_argument("-t", "--threshold", required=True, help="confidence threshold value")

    
    args = vars(parser.parse_args())

    detect_video(videoPath=args["input"], outputVideoPath=args["output"], modelFile=args["model"], threshold=float(args["threshold"]))





