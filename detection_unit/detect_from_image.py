# Import dependencies

import time
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

import torchvision

import argparse

from image_detector import detect_image

def detect_from_image(imagePath, modelPath, threshold):
    start_time = time.time()
    out_image = detect_image(image=Image.open(imagePath), num_classes=4,model_file=modelPath, show_image=False, threshold=threshold)
    print("Detected in ",time.time()-start_time,"seconds")
    img = np.asarray(out_image[0])
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


if __name__ == "__main__":
    
    # initiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to input image file")
    parser.add_argument("-m", "--model", required=True, help="path to model file")
    parser.add_argument("-t", "--threshold", required=True, help="confidence threshold value")

    
    args = vars(parser.parse_args())
    detect_from_image(args["image"], args["model"], float(args["threshold"]))
    




