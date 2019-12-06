# Import dependencies

import time
import cv2
import torchvision.transforms.functional as F
import torch
from PIL import Image

import torchvision
import torchvision.models.detection

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from drawing_tools import draw_image_with_boxes

def get_model_instance_segmentation(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = num_classes  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def detect_image(image=None, imagePath=None, num_classes=2, model_file=None, show_image=True, threshold=0.5):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tr = torchvision.transforms.ToTensor()
    if imagePath is not None:
        image = tr(Image.open(imagePath).convert("RGB"))
    else:
        image = tr(image.convert("RGB"))
    #image = trans(Image.open(img_path).convert("RGB"))
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    checkpoint = torch.load(model_file, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    image = [image.to(device)]
    #torch.cuda.synchronize()
    model_time = time.time()
    outputs = model(image)
    outputs = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in outputs]
    model_time = time.time() - model_time
    out_images = []
    for img, output in zip(image, outputs):
        pred_boxes = []
        pred_labels = []
        cpu_image = img.cpu()
        output_boxes = output["boxes"].cpu()
        labels = output["labels"].cpu()
        scores = output["scores"].cpu()
        for box, score, label in zip(output_boxes.tolist(), scores.tolist(), labels.tolist()):
            if (score > threshold):
                pred_boxes.append(box)
                pred_labels.append(label)
        
        out_images.append(draw_image_with_boxes(F.to_pil_image(cpu_image), pred_boxes, pred_labels, show_image=show_image))
        print("Predicted Locations:\n",pred_boxes)
    return out_images


