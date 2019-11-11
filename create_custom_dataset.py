#Author: Nafisa Ali Amir
#Date: Nov 10, 2019
#Description: Download Open images dataset from google for LINDT specific classes namely "Door handle and Light switch" in the format acceptable for yolo v3 transfer learning
#Dataset Structure: Let crd be current directory namely 'darknet/custom_dataset' where 'darknet' is obtained from https://github.com/pjreddie/darknet
# Let custom_dataset contain this script 'create_dataset.py'
#Create two directories as follows:
#   mkdir images
#   mkdir labels
#How to run the script: python3 create_dataset.py
#No inputs or arguments are required
#Output Directory -
# - - - - - - - - - - - - > crd/images/train_<imageID>.jpg
# - - - - - - - - - - - - > crd/images/validation_<imageID>.jpg
# - - - - - - - - - - - - > crd/images/test_<imageID>.jpg
# - - - - - - - - - - - - > crd/labels/train_<imageID>.txt
# - - - - - - - - - - - - > crd/labels/validation_<imageID>.txt
# - - - - - - - - - - - - > crd/labels/test_<imageID>.txt
#Rewrites ../data/coco.names
import requests
import csv
import urllib.request
import codecs
import os

#globals
subsets = ['train', 'val', 'test']
classNames = ['Door handle','Light switch']

MIDs = 'https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv'

annotations = {}
annotations['train'] = 'https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv'
annotations['val'] = 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv'
annotations['test'] = 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv'


imageURLs = {}
imageURLs['train'] = 'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'
imageURLs['val'] = 'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv'
imageURLs['test'] = 'https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv'

coco_names = '../yolov3/data/coco.names'

def getMIDs(url, classNames):
    labelMID = {}

    ftpstream = urllib.request.urlopen(url)
    csvfile = csv.reader(codecs.iterdecode(ftpstream, 'utf-8'))
    for lineContents in csvfile:
        if lineContents[1] in classNames:
            labelMID[lineContents[1]] = lineContents[0] 

    return labelMID


def getImageIDsFromMID(url, matchMID):

    imageIDs = {}

    ftpstream = urllib.request.urlopen(url)
    csvfile = csv.reader(codecs.iterdecode(ftpstream, 'utf-8'))
    next(csvfile)
    for lineContents in csvfile:
        imageID = lineContents[0]
        MID = lineContents[2]
        [left, right, top, bot] = [float(lineContents[4]), float(lineContents[5]), float(lineContents[6]), float(lineContents[7])]
        if MID == matchMID:
            imageIDs[imageID] = [left, right, top, bot]

    return imageIDs


def getImageURLsFromIDs(url, imageIDs):
    
    ftpstream = urllib.request.urlopen(url)
    csvfile = csv.reader(codecs.iterdecode(ftpstream, 'utf-8'))
    next(csvfile)
    for lineContents in csvfile:
        imageID = lineContents[0]
        imageURL = lineContents[2]
        if imageID in imageIDs:
            imageIDs[imageID].append(imageURL)

    return imageIDs 

def downloadImages(subset, classNameIndex, imageIDs):
    for imageID, [left, right, top, bot, imageURL] in imageIDs.items():

        filePath = 'images/'+subset+'2014/'+imageID+'.jpg'
        with open(filePath, 'wb') as handle:
            response = requests.get(imageURL, stream=True)

            if not response.ok:
                print(response)
                print("Removing "+filePath)
                os.remove(filePath)
                continue
            for block in response.iter_content(1024):
                if not block:
                    break

                handle.write(block)
        print("Downloaded "+imageID+" from "+imageURL)
                
        x_center = str(left + ((right - left)/2))
        y_center = str(top + ((bot - top)/2))
        width = str(right - left)
        height = str(bot - top)
        classNameIndex = str(classNameIndex)

        with open('labels/'+subset+'2014/'+imageID+'.txt', 'w') as file:
            file.write(classNameIndex+' '+x_center+' '+y_center+' '+width+' '+height+'\n')

def rewrite_coco_names(file):
    with open(file, "w") as f:
        for className in classNames:
            f.write(className+'\n')

def create_directory_structure(directories):
   
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def sanity_check(subset, imageIDs):
    for imageID in imageIDs.keys():

        filePath = 'images/'+subset+'2014/'+imageID+'.jpg'
        if (os.path.exists(filePath)):
            if os.path.getsize(filePath) == 0:
                print(filePath)
                os.remove(filePath)

if __name__ == "__main__":
    directories = ['images','images/train2014/','images/val2014','images/test2014','labels','labels/train2014','labels/val2014','labels/test2014']
    create_directory_structure(directories)

    labelMID = getMIDs(MIDs, classNames)
    for classNameIndex, className in enumerate(classNames):
        for subset in subsets:
            print("Working on " + className + " for " + subset)
            imagesData = getImageIDsFromMID(annotations[subset], labelMID[className])
            imagesData = getImageURLsFromIDs(imageURLs[subset], imagesData)
            downloadImages(subset, classNameIndex, imagesData)
            sanity_check(subset, imagesData)

    rewrite_coco_names(coco_names)

