# code/mmdetection_trash/mmdet/datasets/pipelines/my_pipeline.py

from pycocotools.coco import COCO
from mmdet.datasets import PIPELINES
from mmdet.core.mask import BitmapMasks

import os
import cv2
import numpy as np
from numpy import random
import pycocotools.mask as maskUtils


@PIPELINES.register_module()
class MyTransform:
    def __init__(self, mosaic_ratio=0.5):
        self.mosaic_ratio = mosaic_ratio
        
        self.data_dir = '/opt/ml/input/data'
        json = os.path.join(self.data_dir, 'train.json')
        cateory_names = ['Battery', 'UNKNOWN', 'Clothing']
        self.coco = COCO(json)
        self.imgIds = self.get_imgIds(cateory_names)
        self.imgs = self.coco.loadImgs(self.imgIds)
        
        
    def get_imgIds(self, category_names):
        imgIds = []
        for category_name in category_names:
            catIds = self.coco.getCatIds(catNms=[category_name])
            imgIds.extend(self.coco.getImgIds(catIds=catIds))

        return imgIds
        
        
    def make_mosaic_img(self, ori_img):
        s = 512
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        
        img_infos = []
        for _ in range(3):
            img_infos.append(random.choice(self.imgs))
        img1 = cv2.imread(os.path.join(self.data_dir, img_infos[0]['file_name']))
        img2 = cv2.imread(os.path.join(self.data_dir, img_infos[1]['file_name']))
        img3 = cv2.imread(os.path.join(self.data_dir, img_infos[2]['file_name']))
        
        imgs = [[img1, img_infos[0]['id']], [img2, img_infos[1]['id']], [img3, img_infos[2]['id']], [ori_img, -1]]
        random.shuffle(imgs)
        imgIds = [imgs[0][1], imgs[1][1], imgs[2][1], imgs[3][1]]

        x1a, x2a = (0, 512)
        y1a, y2a = (0, 512)
        mosaic_img[y1a:y2a, x1a:x2a] = imgs[0][0]

        x1a, x2a = (512, 1024)
        y1a, y2a = (0, 512)
        mosaic_img[y1a:y2a, x1a:x2a] = imgs[1][0]

        x1a, x2a = (0, 512)
        y1a, y2a = (512, 1024)
        mosaic_img[y1a:y2a, x1a:x2a] = imgs[2][0]

        x1a, x2a = (512, 1024)
        y1a, y2a = (512, 1024)
        mosaic_img[y1a:y2a, x1a:x2a] = imgs[3][0]
        
        return mosaic_img, imgIds
    
    
    def make_mosaic_bboxes(self, ori_bboxes, ori_labels, imgIds):
        anns_sels = []
        for imgId in imgIds:
            if imgId == -1:
                anns_sels.append([-1])
            else:
                annIds = self.coco.getAnnIds(imgIds=imgId, iscrowd=None)
                anns_sels.append(self.coco.loadAnns(annIds))  # 3차원 배열
            
        mosaic_bboxes = []
        mosaic_labels = []
        for i, anns_sel in enumerate(anns_sels):
            if anns_sel == [-1]:
                for bbox, label in zip(ori_bboxes, ori_labels):
                    # xmin, ymin, xmax, ymax
                    xmin, ymin, xmax, ymax = bbox
                    if i == 1:
                        xmin = xmin+512.
                        xmax = xmax+512.
                    elif i == 2:
                        ymin = ymin+512.
                        ymax = ymax+512
                    elif i == 3:
                        xmin, ymin = (xmin+512., ymin+512.)
                        xmax, ymax = (xmax+512., ymax+512.)
                    mosaic_bboxes.append([xmin, ymin, xmax, ymax])
                    mosaic_labels.append(label)
                
            else:
                for ann_sel in anns_sel:
                    bbox = ann_sel['bbox']
                    label = ann_sel['category_id']
                    # xmin, ymin, width, height -> xmin, ymin, xmax, ymax
                    xmin, ymin, width, height = bbox
                    if i == 1:
                        xmin = xmin+512.
                    elif i == 2:
                        ymin = ymin+512.
                    elif i == 3:
                        xmin, ymin = (xmin+512., ymin+512.)
                    mosaic_bboxes.append([xmin, ymin, xmin+width, ymin+height])
                    mosaic_labels.append(label)
                
        mosaic_bboxes = np.array(mosaic_bboxes, dtype=np.float32)
        mosaic_labels = np.array(mosaic_labels)
        return mosaic_bboxes, mosaic_labels
    
    
    def annToRLE(self, segm):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        h, w = 512, 512
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle


    def annToMask(self, segm):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(segm)
        m = maskUtils.decode(rle)
        return m


    def make_mosaic_masks(self, ori_masks, imgIds):
        anns_sels = []
        for imgId in imgIds:
            if imgId == -1:
                anns_sels.append([-1])
            else:
                annIds = self.coco.getAnnIds(imgIds=imgId, iscrowd=None)
                anns_sels.append(self.coco.loadAnns(annIds))  # 3차원 배열
                
        mosaic_masks = []
        s = 512
        for i, anns_sel in enumerate(anns_sels):
            if anns_sel == [-1]:
                for ori_mask in ori_masks:
                    mosaic_mask = np.full((s * 2, s * 2), 0, dtype=np.uint8)
                    
                    if i == 0:
                        x1a, x2a = (0, 512)
                        y1a, y2a = (0, 512)
                    elif i == 1:
                        x1a, x2a = (512, 1024)
                        y1a, y2a = (0, 512)
                    elif i == 2:
                        x1a, x2a = (0, 512)
                        y1a, y2a = (512, 1024)
                    elif i == 3:
                        x1a, x2a = (512, 1024)
                        y1a, y2a = (512, 1024)
                        
                    mosaic_mask[y1a:y2a, x1a:x2a] = ori_mask
                    mosaic_masks.append(mosaic_mask)
                
            else:
                for ann_sel in anns_sel:
                    mosaic_mask = np.full((s * 2, s * 2), 0, dtype=np.uint8)
                    segm = ann_sel['segmentation']
                    mask = self.annToMask(segm)
                    
                    if i == 0:
                        x1a, x2a = (0, 512)
                        y1a, y2a = (0, 512)
                    elif i == 1:
                        x1a, x2a = (512, 1024)
                        y1a, y2a = (0, 512)
                    elif i == 2:
                        x1a, x2a = (0, 512)
                        y1a, y2a = (512, 1024)
                    elif i == 3:
                        x1a, x2a = (512, 1024)
                        y1a, y2a = (512, 1024)
                        
                    mosaic_mask[y1a:y2a, x1a:x2a] = mask
                    mosaic_masks.append(mosaic_mask)
                    
        mosaic_masks = BitmapMasks(mosaic_masks, s * 2, s * 2)
        return mosaic_masks
            

    def __call__(self, results):
        """Call function to random mosaic images, bounding boxes, masks.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: mosaiced results.
        """
        if random.random() < self.mosaic_ratio:
            ori_img = results['img']
            ori_bboxes = results['gt_bboxes']
            ori_labels = results['gt_labels']
                    
            results['img'], imgIds = self.make_mosaic_img(ori_img)
            results['gt_bboxes'], results['gt_labels'] = self.make_mosaic_bboxes(ori_bboxes, ori_labels, imgIds)
            
            if 'gt_masks' in results.keys():
                ori_masks = results['gt_masks']
                results['gt_masks'] = self.make_mosaic_masks(ori_masks, imgIds)
                
        return results