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
import requests
import csv
import urllib.request
import codecs
import os
from tqdm import tqdm

def create_directory_structure(directories):
   
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def sanity_check(imageIDs):
    for subset, info in imageIDs.items():
        for imageID in info.keys():
            filePath = os.path.join(subset,'images',imageID+'.jpg')
            labelfilePath = os.path.join(subset,'labels',imageID+'.txt')
            if (os.path.exists(filePath)):
                if os.path.getsize(filePath) == 0:
                    print(filePath)
                    os.remove(filePath)
                    if (os.path.exists(labelfilePath)):
                        print(labelfilePath)
                        os.remove(labelfilePath)


def downloadImages(imageInfo):
    for subset, details in imageInfo.items():
        total_downloads = 1
        n_downloads = [0, 0, 0]
        for imageID, info in details.items():
            if n_downloads[info['labels'][0][0]] < 600:
                n_downloads[info['labels'][0][0]] = n_downloads[info['labels'][0][0]] + 1

                filePath = os.path.join(subset,'images',imageID+'.jpg')
                with open(filePath, 'wb') as handle:
                    response = requests.get(info['url'], stream=True)
                    if not response.ok:
                        print(response)
                        print("Removing "+filePath)
                        os.remove(filePath)
                        continue
                    for block in response.iter_content(1024):
                        if not block:
                            break

                        handle.write(block)
            
                print(total_downloads,"Downloaded "+imageID+" from "+info['url'])
                filePath = os.path.join(subset,'labels',imageID+'.txt')
                with open(filePath,'w') as file:
                    for [category, x_min, y_min, x_max, y_max] in info['labels']:
                        file.write(str(category)+' '+str(x_min)+' '+str(y_min)+' '+str(x_max)+' '+str(y_max)+'\n')
                        idx = category

                print("Individual Downloads:",n_downloads)
                total_downloads += 1



def extractImageURLsFromIDs(URLs, imageIDs):
    print("Extracting Image URLs")
    for subset, url in URLs.items():
        print("Working on subset: ",subset)
        ftpstream = urllib.request.urlopen(url)
        csvfile = csv.reader(codecs.iterdecode(ftpstream, 'utf-8'))
        next(csvfile)
        for lineContents in tqdm(csvfile):
            imageID = lineContents[0]
            imageURL = lineContents[2]
            if imageID in imageIDs[subset]:
                imageIDs[subset][imageID]['url'] = imageURL
    print("Done.")
    return imageIDs 

def extractImageIDs(URLs, categories):
    print("Extracting Image Ids...")
    imageIDs = {}
    

    for subset, url in URLs.items():
        print("Working on subset: ",subset)
        if subset not in imageIDs:
            imageIDs[subset] = {}

        ftpstream = urllib.request.urlopen(url)
        csvfile = csv.reader(codecs.iterdecode(ftpstream, 'utf-8'))
        next(csvfile)
        for lineContents in tqdm(csvfile):
            imageID = lineContents[0]
            category = lineContents[2]
            [x_min, x_max, y_min, y_max] = [float(lineContents[4]), float(lineContents[5]), float(lineContents[6]), float(lineContents[7])]
            if category in categories:
                if imageID not in imageIDs[subset]:
                    imageIDs[subset][imageID] = {}
                    imageIDs[subset][imageID]['labels'] = []
                imageIDs[subset][imageID]['labels'].append([categories.index(category), x_min, y_min, x_max, y_max])

    print("Done.")
    return imageIDs

def extractMIDs(url, objects):
    print("Extracting MIDs...")
    categories = []

    ftpstream = urllib.request.urlopen(url)
    csvfile = csv.reader(codecs.iterdecode(ftpstream, 'utf-8'))
    for lineContents in tqdm(csvfile):
        if lineContents[1] in objects:
            categories.append(lineContents[0])
    
    print("Done.")
    return categories 


if __name__ == "__main__":

    directories = ['train','val','test','train/images','val/images','test/images','train/labels','val/labels','test/labels']
    
    classNames = ["Door handle, Person, Light switch"]

    MID_url = 'https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv'
    annotations = {}
    annotations['train'] = 'https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv'
    annotations['val'] = 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv'
    annotations['test'] = 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv'

    imageURLs = {}
    imageURLs['train'] = 'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'
    imageURLs['val'] = 'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv'
    imageURLs['test'] = 'https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv'



    create_directory_structure(directories)
    categories = extractMIDs(MID_url, classNames)
    images_info = extractImageIDs(annotations, categories)
    images_info = extractImageURLsFromIDs(imageURLs, images_info)
    downloadImages(images_info)
    sanity_check(images_info)

