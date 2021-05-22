# mmdetection_trash/mmdet/datasets/pipelines/my_pipeline.py
# https://mmdetection.readthedocs.io/en/latest/tutorials/data_pipeline.html

from pycocotools.coco import COCO
from mmdet.datasets import PIPELINES
from mmdet.core.mask import BitmapMasks

import os
import cv2
import numpy as np
from numpy import random
import pycocotools.mask as maskUtils


@PIPELINES.register_module()
class Mosaic:
    """Center crop 후 Mosaic 4 images & bbox & mask.

    Args:
        mosaic_ratio (float, optional): mosaic이 될 확률. Defaults to 0.5.
        center_xmin (int, optional): Crop될 부분의 left. Defaults to 70.
        center_ymin (int, optional): Crop될 부분의 bottom. Defaults to 70.
        center_xmax (int, optional): Crop될 부분의 top. Defaults to 450.
        center_ymax (int, optional): Crop될 부분의 right. Defaults to 450.
    """       
    def __init__(self, mosaic_ratio=0.5, center_xmin=70, center_ymin=70, center_xmax=450, center_ymax=450):
        self.mosaic_ratio = mosaic_ratio
        
        self.data_dir = '/opt/ml/input/data'
        json = os.path.join(self.data_dir, 'train.json')    # 추가될 이미지를 뽑을 json file
        cateory_names = ['Battery', 'UNKNOWN', 'Clothing']  # 추가될 이미지에 있길 원하는 category 리스트
        self.coco = COCO(json)
        self.imgIds = self.get_imgIds(cateory_names)
        self.imgs = self.coco.loadImgs(self.imgIds)
        
        self.center_xmin = center_xmin
        self.center_ymin = center_ymin
        self.center_xmax = center_xmax
        self.center_ymax = center_ymax
        
        self.center_width = center_xmax-center_xmin
        self.center_height = center_ymax-center_ymin
        
        
    def get_imgIds(self, category_names):
        """원하는 Cateogry가 존재하는 이미지의 Id들 얻기

        Args:
            category_names (list): 추가될 이미지에 있길 바라는 Category 리스트

        Returns:
            (list): 원하는 Category가 존재하는 이미지들의 Id
        """        
        imgIds = []
        for category_name in category_names:
            catIds = self.coco.getCatIds(catNms=[category_name])
            imgIds.extend(self.coco.getImgIds(catIds=catIds))

        return imgIds
        
        
    def make_mosaic_img(self, ori_img):
        """원래의 이미지에 3개의 이미지를 추가한 mosaic된 이미지 생성

        Args:
            ori_img (ndarray): 원래의 이미지

        Returns:
            (ndarray[uint8], list): mosaic된 이미지, mosaic된 이미지의 id(top-left, top-right, bottom-left, bottom-right 이미지 순서)
        """        
        w = self.center_width
        h = self.center_height
        center_xmin, center_xmax, center_ymin, center_ymax = (self.center_xmin, self.center_xmax, self.center_ymin, self.center_ymax)
        mosaic_img = np.full((h * 2, w * 2, 3), 114, dtype=np.uint8)  # H x W x C
        
        # 랜덤하게 3개의 image 추출
        img_infos = []
        for _ in range(3):
            img_infos.append(random.choice(self.imgs))
        
        img1 = cv2.imread(os.path.join(self.data_dir, img_infos[0]['file_name']))
        img2 = cv2.imread(os.path.join(self.data_dir, img_infos[1]['file_name']))
        img3 = cv2.imread(os.path.join(self.data_dir, img_infos[2]['file_name']))
        
        imgs = [[img1, img_infos[0]['id']], [img2, img_infos[1]['id']], [img3, img_infos[2]['id']], [ori_img, -1]]
        random.shuffle(imgs)  # 원래의 이미지의 위치가 고정되지 않도록 shuffle
        imgIds = [imgs[0][1], imgs[1][1], imgs[2][1], imgs[3][1]]

        x1a, x2a = (0, w)
        y1a, y2a = (0, h)
        croped_img = imgs[0][0][center_ymin:center_ymax, center_xmin:center_xmax, :]  # Center crop
        mosaic_img[y1a:y2a, x1a:x2a] = croped_img

        x1a, x2a = (w, w*2)
        y1a, y2a = (0, h)
        croped_img = imgs[1][0][center_ymin:center_ymax, center_xmin:center_xmax, :]
        mosaic_img[y1a:y2a, x1a:x2a] = croped_img

        x1a, x2a = (0, w)
        y1a, y2a = (h, h*2)
        croped_img = imgs[2][0][center_ymin:center_ymax, center_xmin:center_xmax, :]
        mosaic_img[y1a:y2a, x1a:x2a] = croped_img

        x1a, x2a = (w, w*2)
        y1a, y2a = (h, h*2)
        croped_img = imgs[3][0][center_ymin:center_ymax, center_xmin:center_xmax, :]
        mosaic_img[y1a:y2a, x1a:x2a] = croped_img
        
        return mosaic_img, imgIds    
        

    # https://mmdetection.readthedocs.io/en/latest/_modules/mmdet/datasets/pipelines/transforms.html
    def _filter_boxes(self, patch, boxes):
        """Check whether the center of each box is in the patch.

        Args:
            patch (list[int]): , [left, top, right, bottom].
            boxes (numpy array, (N x 4)): Ground truth boxes.

        Returns:
            mask (numpy array, (N,)): Each box is inside or outside the patch.
        """
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask = (center[:, 0] >= patch[0]) * (center[:, 1] >= patch[1]) * (center[:, 0] <= patch[2]) * (center[:, 1] <= patch[3])
        return mask
    
    
    def make_mosaic_bboxes(self, ori_bboxes, ori_labels, imgIds):
        """원래의 이미지에 3개의 이미지를 추가한 mosaic된 이미지에 대한 bbox 생성

        Args:
            ori_bboxes (ndarray): 원래의 이미지의 bboxes.
            ori_labels (ndarray): 원래의 이미지의 labels.
            imgIds (list): mosaic된 이미지의 위치별 이미지 Id, 원래의 이미지의 id는 -1.

        Returns:
            ndarray, ndarray, list: mosaic된 이미지의 bboxes, mosaic된 이미지의 labels, 사용된 bbox를 표시하는 mask
        """        
        w = self.center_width
        h = self.center_height
        center_xmin, center_xmax, center_ymin, center_ymax = (self.center_xmin, self.center_xmax, self.center_ymin, self.center_ymax)
        
        # 추가된 이미지의 annotation 정보 가져오기
        anns_sels = []
        for imgId in imgIds:
            if imgId == -1:
                anns_sels.append([-1])
            else:
                annIds = self.coco.getAnnIds(imgIds=imgId, iscrowd=None)
                anns_sels.append(self.coco.loadAnns(annIds))  # 3차원 배열
            
        # mosaic된 이미지의 bbox, label 생성
        mosaic_bboxes = []
        mosaic_labels = []
        box_masks = []
        for i, anns_sel in enumerate(anns_sels):
            if anns_sel == [-1]:
                patch = [center_xmin, center_ymin, center_xmax, center_ymax]
                # center crop에 의해, 원래 bbox의 1/4보다 작아진 bboxes 필터링
                mask = self._filter_boxes(patch, ori_bboxes)
                box_masks.append(list(mask))
                ori_bboxes = ori_bboxes[mask]

                # center crop에 의해, bboxes의 위치 변경
                ori_bboxes[:, 0::2] = ori_bboxes[:, 0::2]-center_xmin
                ori_bboxes[:, 1::2] = ori_bboxes[:, 1::2]-center_ymin
                ori_bboxes[:, 0::2] = np.clip(ori_bboxes[:, 0::2], 0, center_xmax)
                ori_bboxes[:, 1::2] = np.clip(ori_bboxes[:, 1::2], 0, center_ymax)

                # mosaic에 의해, bboxes의 위치 변경
                for bbox, label in zip(ori_bboxes, ori_labels):
                    xmin, ymin, xmax, ymax = bbox
                    if i == 1:
                        xmin = xmin+w
                        xmax = xmax+w
                    elif i == 2:
                        ymin = ymin+h
                        ymax = ymax+h
                    elif i == 3:
                        xmin, ymin = (xmin+w, ymin+h)
                        xmax, ymax = (xmax+w, ymax+h)
                    mosaic_bboxes.append([xmin, ymin, xmax, ymax])
                    mosaic_labels.append(label)
            
            # coco data set: xmin, ymin, width, height -> results: xmin, ymin, xmax, ymax
            else:
                mask = []
                for ann_sel in anns_sel:
                    bbox = ann_sel['bbox']
                    label = ann_sel['category_id']
                    xmin, ymin, width, height = bbox
                    xmax, ymax = (xmin+width, ymin+height)

                    # center crop에 의해, 원래 bbox의 1/4보다 작아진 bboxes 필터링
                    center_x, center_y = (xmin + width/2, ymin + height/2)
                    if not ((center_x >= center_xmin and center_x <= center_xmax) and (center_y >= center_ymin and center_y <=center_ymax)):
                        mask.append(False)
                        continue

                    # center crop에 의해, bboxes의 위치 변경
                    mask.append(True)
                    xmin, ymin, xmax, ymax = xmin-center_xmin, ymin-center_ymin, xmax-center_xmin, ymax-center_ymin
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(xmax, w)
                    ymax = min(ymax, h)
                        
                    # mosaic에 의해, bboxes의 위치 변경
                    if i == 1:
                        xmin = xmin+w
                        xmax = xmax+w
                    elif i == 2:
                        ymin = ymin+h
                        ymax = ymax+h
                    elif i == 3:
                        xmin, ymin = (xmin+w, ymin+h)
                        xmax, ymax = (xmax+w, ymax+h)
                    mosaic_bboxes.append([xmin, ymin, xmax, ymax])
                    mosaic_labels.append(label)
                box_masks.append(mask)
            
        mosaic_bboxes = np.array(mosaic_bboxes, dtype=np.float32)
        mosaic_labels = np.array(mosaic_labels)
        return mosaic_bboxes, mosaic_labels, box_masks
    
    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
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


    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
    def annToMask(self, segm):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(segm)
        m = maskUtils.decode(rle)
        return m


    def make_mosaic_masks(self, ori_masks, imgIds, box_masks):
        """원래의 이미지에 3개의 이미지를 추가한 mosaic된 이미지의 mask 생성

        Args:
            ori_masks (BitmapMask): 원래의 이미지의 mask
            imgIds (list): mosaic된 이미지의 위치별 이미지 Id, 원래의 이미지의 id는 -1.
            box_masks (list): center crop 이후 사용된 bbox를 표시하는 mask

        Returns:
            BitmapMasks: mosaic된 이미지의 masks
        """
        w = self.center_width
        h = self.center_height
        center_xmin, center_xmax, center_ymin, center_ymax = (self.center_xmin, self.center_xmax, self.center_ymin, self.center_ymax)
        
        # 추가된 이미지의 annotation 정보 가져오기
        anns_sels = []
        for imgId in imgIds:
            if imgId == -1:
                anns_sels.append([-1])
            else:
                annIds = self.coco.getAnnIds(imgIds=imgId, iscrowd=None)
                anns_sels.append(self.coco.loadAnns(annIds))  # 3차원 배열
                
        # mosaic된 이미지의 mask 생성
        mosaic_masks = []
        s = 512
        for i, (anns_sel, box_mask) in enumerate(zip(anns_sels, box_masks)):
            if anns_sel == [-1]:
                for ori_mask, isin_center in zip(ori_masks, box_mask):
                    if not isin_center:     # 해당 mask와 관련된 bbox가 center crop에 의해 사라지지 않았는지 확인
                        continue
                    mosaic_mask = np.full((h * 2, w * 2), 0, dtype=np.uint8)
                    
                    if i == 0:
                        x1a, x2a = (0, w)
                        y1a, y2a = (0, h)
                    elif i == 1:
                        x1a, x2a = (w, w*2)
                        y1a, y2a = (0, h)
                    elif i == 2:
                        x1a, x2a = (0, w)
                        y1a, y2a = (h, h*2)
                    elif i == 3:
                        x1a, x2a = (w, w*2)
                        y1a, y2a = (h, h*2)
                        
                    mosaic_mask[y1a:y2a, x1a:x2a] = ori_mask[center_ymin:center_ymax, center_xmin:center_xmax]
                    mosaic_masks.append(mosaic_mask)
                
            else:
                for ann_sel, isin_center in zip(anns_sel, box_mask):
                    if not isin_center:     # 해당 mask와 관련된 bbox가 center crop에 의해 사라지지 않았는지 확인
                        continue
                    mosaic_mask = np.full((h * 2, w * 2), 0, dtype=np.uint8)
                    segm = ann_sel['segmentation']
                    mask = self.annToMask(segm)
                    
                    if i == 0:
                        x1a, x2a = (0, w)
                        y1a, y2a = (0, h)
                    elif i == 1:
                        x1a, x2a = (w, w*2)
                        y1a, y2a = (0, h)
                    elif i == 2:
                        x1a, x2a = (0, w)
                        y1a, y2a = (h, h*2)
                    elif i == 3:
                        x1a, x2a = (w, w*2)
                        y1a, y2a = (h, h*2)
                        
                    mosaic_mask[y1a:y2a, x1a:x2a] = mask[center_ymin:center_ymax, center_xmin:center_xmax]
                    mosaic_masks.append(mosaic_mask)
                    
        mosaic_masks = BitmapMasks(mosaic_masks, h * 2, w * 2)
        return mosaic_masks
            

    def __call__(self, results):
        """Call function to random mosaic images, bounding boxes, masks.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: mosaic results.
        """
        if random.random() < self.mosaic_ratio:
            ori_img = results['img']
            ori_bboxes = results['gt_bboxes']
            ori_labels = results['gt_labels']
                    
            results['img'], imgIds = self.make_mosaic_img(ori_img)
            results['gt_bboxes'], results['gt_labels'], box_masks = self.make_mosaic_bboxes(ori_bboxes, ori_labels, imgIds)
            
            if 'gt_masks' in results.keys():
                ori_masks = results['gt_masks']
                results['gt_masks'] = self.make_mosaic_masks(ori_masks, imgIds, box_masks)
                
        return results
    
    
    
    
@PIPELINES.register_module()
class Mixup:
    def __init__(self, mixup_ratio=0.5, alpha=1.0):
        self.mixup_ratio = mixup_ratio
        self.alpha = alpha
        
        self.data_dir = '/opt/ml/input/data'
        json = os.path.join(self.data_dir, 'train.json')
        self.coco = COCO(json)
        self.imgIds = self.coco.getImgIds()
        
    
    # https://www.kaggle.com/kaushal2896/data-augmentation-tutorial-basic-cutout-mixup
    def mixup(self, ori_img, ori_bboxes, ori_labels):
        """원래의 이미지와 랜덤으로 뽑힌 이미지를 mixup.

        Args:
            ori_img (ndarray): 원래의 이미지.
            ori_bboxes (ndarray): 원래의 이미지의 bboxes.
            ori_labels (ndarray): 원래의 이미지의 labels.

        Returns:
            ndarray, ndarray, ndarray: mixup된 이미지, mixup된 이미지의 bboxes, mixup된 이미지의 labels
        """        
        
        alpha = self.alpha
        
        # 원래의 이미지와 mixup될 이미지 랜덤하게 추출
        imgId = random.choice(self.imgIds)
        imgInfos = self.coco.loadImgs([imgId])
        annIds = self.coco.getAnnIds(imgIds=imgId, iscrowd=None)
        
        choosed_img = cv2.imread(os.path.join(self.data_dir, imgInfos[0]['file_name']))
        choosed_anns = self.coco.loadAnns(annIds)
        
        
        # Generate image weight (minimum 0.4 and maximum 0.6)
        lam = np.clip(np.random.beta(alpha, alpha), 0.4, 0.6)

        # Weighted Mixup
        mixup_img = lam*ori_img + (1 - lam)*choosed_img

        # 원래의 이미지의 bboxes와 labels에 추가된(mixup된) 이미지의 bboxes와 labels 정보 추가
        mixup_bboxes = list(ori_bboxes)
        mixup_lables = list(ori_labels)
        for choosed_ann in choosed_anns:
            mixup_bboxes.append(choosed_ann['bbox'])
            mixup_lables.append(choosed_ann['category_id'])

        mixup_bboxes = np.array(mixup_bboxes, dtype=np.float32)
        mixup_lables = np.array(mixup_lables)

        return mixup_img, mixup_bboxes, mixup_lables
    
    
    def __call__(self, results):
        """Call function to random mixup images, bounding boxes, masks.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: mixup results.
        """
        if random.random() < self.mixup_ratio:
            ori_img = results['img']
            ori_bboxes = results['gt_bboxes']
            ori_labels = results['gt_labels']
                    
            results['img'], results['gt_bboxes'], results['gt_labels'] = self.mixup(ori_img, ori_bboxes, ori_labels)
            
            # segmentation 정보를 필요로 하는 경우 사용 제한
            if 'gt_masks' in results.keys():
                raise RuntimeError("masks can't use") 
                    
        return results
