from mmdet.datasets import (build_dataloader, build_dataset)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from pycocotools.coco import COCO

import os
import cv2
import wandb
import numpy as np
import matplotlib.pyplot as plt


classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
        "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
class_id_to_label = {v : k for v, k in enumerate(classes)}
class_set = wandb.Classes([{'name': name, 'id': id} for id, name in class_id_to_label.items()])
class_num = 11
dataset_dir = '/opt/ml/input/data'


def get_data_and_output(cfg, checkpoint_path, mode):
    """사용한 data와 해당 data에 대한 모델의 추론 결과 반환

    Args:
        cfg (Config): 학습에 사용한 config 파일
        checkpoint_path (str): 학습에 사용한 모델의 checkpoint가 저장된 경로
        mode (str): 사용할 dataset의 종류. validation dataset('val'), test dataset('test') 중 하나

    Raises:
        ValueError: mode 값으로 'val', 'test'가 아닌 문자열이 들어올 경우 error 발생

    Returns:
        data, output: 사용한 data, 해당 data에 대한 모델의 추론 결과
    """    
    if mode == "test":
        data = cfg.data.test
    elif mode == "val":
        data = cfg.data.val
    else:
        raise ValueError('mode can have val or test')

    dataset = build_dataset(data)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')

    model.CLASSES = classes
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader, show_score_thr=0.05)

    return data, output


# https://wandb.ai/stacey/yolo-drive/reports/Bounding-Boxes-for-Object-Detection--Vmlldzo4Nzg4MQ
def bounding_boxes(filename, boxes, labels, scores=None, mode="ground_truth"):
    # load raw input photo
    raw_image = cv2.imread(filename, cv2.IMREAD_COLOR)
    all_boxes = []
    # plot each bounding box for this image
    for b_i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = map(float, box)
        # get coordinates and labels
        box_data = {"position" : {
          "minX" : xmin,
          "maxX" : xmax,
          "minY" : ymin,
          "maxY" : ymax},
          "class_id" : labels[b_i],
          "domain" : "pixel"}
        
        if (mode == "predictions") and (scores != None):
            box_data["scores"] = {"score" : float(scores[b_i])}
        elif mode == "ground_truth":
            box_data["scores"] = {"score" : 1}
            
        all_boxes.append(box_data)

    # log to wandb: raw image, predictions, and dictionary of class labels for each class id
    box_image = wandb.Image(raw_image, classes=class_set,
                            boxes={mode: {"box_data": all_boxes, "class_labels" : class_id_to_label}})
    return box_image


def get_image(filename, p_boxes, p_labels, p_scores, g_boxes, g_labels):
    """Wandb에서 사용가능한 이미지로 변경

    Args:
        filename (str): 추론에 사용된 이미지 파일 이름
        p_boxes (list): 추론된 bbox 리스트
        p_labels (list): 추론된 label 리스트
        p_scores (list): 추론된 cateogry score 리스트
        g_boxes (ndarray): 정답 bbox 리스트
        g_labels (list): 정답 label 리스트

    Returns:
        Image, Image, Image: 원본 이미지, 예측된 bbox가 그려진 이미지, 정답 bbox가 그려진 이미지
    """    
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
        
    raw_img = wandb.Image(img)
    p_box_img = bounding_boxes(filename, p_boxes, p_labels, p_scores, mode="predictions")
    g_box_img = bounding_boxes(filename, g_boxes, g_labels, mode="ground_truth")
    
    return (raw_img, p_box_img, g_box_img)


def push_image(cfg, checkpoint_path, wandb_name=None, img_num=10, wandb_finish=True):
    """모델의 validation, test data에 대한 추론 결과 image를 Wandb Runs의 media에 전송

    Args:
        cfg (Config): 학습에 사용한 config 파일
        checkpoint_path (str): 학습에 사용한 모델의 checkpoint가 저장된 경로
        wandb_name (str): Runs에 표시되는 이름.
        img_num (int, optional): 시각화할 이미지의 개수. Defaults to 10.
    """

    cfg.data.samples_per_gpu = 4
    init_kwargs = cfg.log_config.hooks[1].init_kwargs  # 설정한 wandb 정보 가져오기
    run = wandb.init(
        project=init_kwargs['project'],
        entity=init_kwargs['entity'],
        name = wandb_name,
        reinit=True,
        job_type="upload",)
    
    for mode in ['val', 'test']:
        data, output = get_data_and_output(cfg, checkpoint_path, mode)
    
        coco = COCO(data.ann_file)
        # 모델의 추론 결과 bbox 정보 추출 후 Wandb에서 사용가능한 이미지로 변경
        imgs = []
        for i, out in enumerate(output):
            p_labels, p_scores, p_bboxes = [], [], []
            for j in range(class_num):
                for o in out[j]:
                    p_labels.append(j)
                    p_scores.append(o[4])
                    xmin, ymin, xmax, ymax = o[0], o[1], o[2], o[3]
                    p_bboxes.append([xmin, ymin, xmax, ymax])

            img_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
            filename = os.path.join(dataset_dir, img_info['file_name'])        
            img_ann = coco.loadAnns(coco.getAnnIds(imgIds=img_info['id']))
            boxes = np.array([x['bbox'] for x in img_ann])
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            labels = list([x['category_id'] for x in img_ann])

            _, p_box_img, g_box_img = get_image(filename, p_bboxes, p_labels, p_scores, boxes, labels)
            imgs.extend([p_box_img, g_box_img])

            if i>=img_num:
                break

        wandb.log({f"object detection {mode}": imgs})
        

    # 이미지 당 모델이 예측한 Category 분포, object의 max area 조사
    categories, max_areas = [], []
    for i, out in enumerate(output):  # test의 output
        p_labels = []
        max_area = 0
        for j in range(class_num):
            for o in out[j]:
                p_labels.append(j)
                xmin, ymin, xmax, ymax = o[0], o[1], o[2], o[3]

                area = (xmax-xmin) * (ymax-ymin)  # area
                if area > max_area:
                    max_area = area
                    
        categories.append(p_labels)
        max_areas.append(max_area)
    
    # 이미지 당 모델이 예측한 Category 분포 시각화
    fig = plt.figure(figsize=(25, 16))
    ax = fig.add_subplot(111)

    plt.title("Predicted categories", fontsize=30)
    plt.yticks(ticks=range(class_num), labels=classes, fontsize=20)
    plt.xlabel('image id', fontsize=20)

    for i, tmp_categories in enumerate(categories):
        ax.scatter(x=[i]*len(tmp_categories), y=tmp_categories, color='royalblue', alpha=0.6)

    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    categories_name = './categories.png'
    plt.savefig(categories_name, dpi=300)

    # 이미지 당 모델이 예측한 object의 max area 시각화
    fig = plt.figure(figsize=(20, 18))
    plt.title("Predicted max area", fontsize=30)
    plt.xlabel('image id', fontsize=20)
    plt.ylabel('max area', fontsize=20)

    for i, max_are in enumerate(max_areas):
        plt.scatter(x=i, y=max_are, color='royalblue', alpha=0.6)
    max_area_name = './max_are.png'
    plt.savefig(max_area_name, dpi=300)
    
    wandb.log({"Plots": (wandb.Image(categories_name), wandb.Image(max_area_name))})

    if os.path.isfile(categories_name):
        os.remove(categories_name)
    if os.path.isfile(max_area_name):
        os.remove(max_area_name)
    
    if wandb_finish:
        run.finish()
    print('Done.')
