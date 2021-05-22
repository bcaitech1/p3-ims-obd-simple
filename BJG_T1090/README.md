## Object Detection

- print_image.py: COCO dataset에서 이미지를 시각화 하는데 사용되는 함수들을 저장한 파일
- visualization.py: wandb에 모델의 학습결과를 시각화할 수 있도록 wandb에 이미지를 logging하는 함수가 구현된 파일
### augmentation
  - \_init_.py: mmedetection pipeline 생성시 사용되는 초기화 파일, mypipeline.py의 Mosaic class와 Mixup class를 추가
  - mypipeline.py: mmdetection train pipeline에 mosaic, mixup 기법을 추가하여 사용할 수 있도록 구현한 파일


## Semantic Segmentation
- EfficientUnet-Adam.ipynb: 기존 baseline을 수정하여 EfficientUnet을 사용하는 코드
- EfficientUnet-swa.ipynb: 기존 baseline을 수정하여 EfficientUnet에 swa를 사용하는 코드
- PSPNet.ipynb: 기존 baseline을 수정하여 PSPNet을 사용하는 코드
- dataset.py: 노트북에서 코드를 줄일 수 있도록 dataset, dataloader 구현 부분을 옮겨둔 파일
- utils.py: 노트북에서 코드를 줄일 수 있도록 시간 정보를 얻을 수 있는 함수를 구현한 파일
- resnet.py: PSPNet에서 사용되는 resnet을 구현한 파일(출처: Pytorch)
