import os
import numpy as np
import torch
from PIL import Image

import transforms as T

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate
import utils

class CustomDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(self.root, "labels"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        labels_path = os.path.join(self.root, "labels", self.labels[idx])
        
        img = Image.open(img_path).convert("RGB")
        (w, h) = img.size

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        #label = Image.open(mask_path)
        boxes, labels = [], []
        with open(labels_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                words = line.strip().split(" ")
                if len(words) == 5:
                    label = int(words[0])+1
                    words = list(map(float, words[1:]))
                    
                    new_words = [min(words[0],words[2]),
                                 min(words[1],words[3]),
                                 max(words[0],words[2]),
                                 max(words[1],words[3])]
                    
                    if ((1 >= new_words[0] >= 0) and
                       (1 >= new_words[1] >= 0) and
                       (1 >= new_words[2] >= 0) and
                       (1 >= new_words[3] >= 0)):
                        new_words[0] = (w)*new_words[0]
                        new_words[1] = (h)*new_words[1]
                        new_words[2] = (w)*new_words[2]
                        new_words[3] = (h)*new_words[3]

                        boxes.append(new_words)
                        labels.append(label)
        
        num_objs = len(boxes)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        #labels = torch.ones((num_objs,), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd


        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


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


def main_train(output_dir=None, resume_file=None, num_epochs=10):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 3 #Modified for door handle, person and background
    # use our dataset and defined transformations
    dataset = CustomDataset('/content/rcnn_dataset/train/', get_transform(train=True))
    dataset_test = CustomDataset('/content/rcnn_dataset/test/', get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    if resume_file:
        checkpoint = torch.load(resume_file, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # let's train it for 10 epochs
    num_epochs = num_epochs

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

        if output_dir:
            if epoch == num_epochs - 1: #if last epoch
                save_model_filePath = os.path.join(output_dir, 'model_last.pth'.format(epoch))
            else:
                save_model_filePath = os.path.join(output_dir, 'model_{}.pth'.format(epoch))
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()},
                save_model_filePath)
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")

if __init__=="__main__":
    main_train(output_dir='output', resume_file=None, num_epochs=15)
