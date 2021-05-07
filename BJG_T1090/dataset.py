import os
import json
import cv2
from pycocotools.coco import COCO

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_path, mode = 'train', transform = None, tta = False):
        super().__init__()
        self.mode = mode
        self.tta = tta
        if not self.tta:
            self.transform = transform
        else:
            self.transform_1 = transform[0]
            self.transform_2 = transform[1]
            self.transform_3 = transform[2]
        self.coco = COCO(data_path)
        self.data_path = os.path.dirname(data_path)
        self.category_names = ['Background', 'UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.data_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
#         images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)
            
            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # number of objects in the image
            num_objs = len(anns)

            # Bounding boxes for objects
            # In coco format, bbox = [xmin, ymin, width, height]
            # In pytorch, the input should be [xmin, ymin, xmax, ymax]
            boxes = []
            for i in range(num_objs):
                xmin = anns[i]['bbox'][0]
                ymin = anns[i]['bbox'][1]
                xmax = xmin + anns[i]['bbox'][2]
                ymax = ymin + anns[i]['bbox'][3]
                boxes.append([xmin, ymin, xmax, ymax])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            
            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # Unknown = 1, General trash = 2, ... , Cigarette = 11
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks = np.maximum(self.coco.annToMask(anns[i])*pixel_value, masks)
            masks = masks.astype(np.float32)
            
            
            # Labels
            labels = torch.ones((num_objs,), dtype=torch.int64)
            
            # Tensorise img_id
            image_id = torch.tensor([image_id])
            
            # Size of bbox (Rectangular)
            areas = []
            for i in range(num_objs):
                areas.append(anns[i]['area'])
            areas = torch.as_tensor(areas, dtype=torch.float32)
            
            # Iscrowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            
            # Annotation is in dictionary format
            image_infos['bbox'] = boxes
            image_infos['labels'] = labels
            image_infos['image_id'] = image_id
            image_infos['area'] = areas
            image_infos['iscrowd'] = iscrowd
            image_infos['file_name'] = image_infos['file_name']

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.tta:
                transformed_1 = self.transform_1(image=images)
                transformed_2 = self.transform_2(image=images)
                transformed_3 = self.transform_3(image=images)
                images_1 = transformed_1["image"]
                images_2 = transformed_2["image"]
                images_3 = transformed_3["image"]
                
                return images_1, images_2, images_3, image_infos
                
            elif self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            
            return images, image_infos
    
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())
    

def get_loader(data_path, batch_size, transform):
    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    # Dataset
    dataset = CustomDataLoader(data_path=data_path, mode='train', transform=transform)

    # DataLoader
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         num_workers=4,
                                         collate_fn=collate_fn)
    
    return loader