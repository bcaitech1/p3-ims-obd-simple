# [P Stage 3] Semantic segmentation & Object detection

## Semantic segmenation
![image](https://user-images.githubusercontent.com/71882533/119220651-7457e200-bb26-11eb-921e-a116cb95a59d.png)

### 개요
- 2021.04.26(월) ~ 05.06(목) 19:00
- 쓰레기가 찍힌 사진에서 쓰레기를 object별로 Segmentation 하는 모델 생성
- 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기가 찍힌 사진 데이터셋이 제공

### 사용한 모델
- Unet
- DeepLab v3 Plus

## Object detection
![image](https://user-images.githubusercontent.com/71882533/119220784-21caf580-bb27-11eb-8e09-2404073f3066.png)

### 개요
- 2021.05.10(월) ~ 05.20(목) 19:00
- 쓰레기가 찍힌 사진에서 쓰레기를 object별로 Detection 하는 모델 생성
- 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기가 찍힌 사진 데이터셋이 제공

### 사용한 모델
- Faster R-CNN `Resnet 50`
- Mask R-CNN `Swin`
- Cascade Mask R-CNN `Swin`
- VfNet `Resnet 101`
- UniverseNet `Res2net 101`

## 데이터
- 전체 이미지 개수 : 4109장
  - train: 2616장
  - validation: 655장
  - test(public): 417장
  - test(private): 420장
- 11 class : UNKNOWN, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : (512, 512)

### 데이터에서 발견된 문제와 문제 해결을 위해 시도한 내용
- Data Imbalance 문제
  <br/><img src = "https://user-images.githubusercontent.com/71882533/119221019-3cea3500-bb28-11eb-8af3-bd91b9d8bb89.png" width="400px" height="350px">
  - Mosaic 기법 사용 `실패`
    Mosaic 기법을 통해 상대적으로 수가 적은 class(UNKNOWN, Clothing, Battery)의 object가 있는 사진 3장을 기존의 이미지에 추가하여 모델에 더 많이 제공
    <br/><img src = "https://user-images.githubusercontent.com/71882533/119221439-6310d480-bb2a-11eb-9f3f-29e6db81e80d.png" width="400px" height="350px">
  - 성능 좋은 모델 사용 `성공`
    현재 Segmentation, Object detection에서 SOTA인 Swin 모델 사용

### 성능 향상을 위해 시도한 내용
- Pseudo Labeling `성공`
- WBF를 통한 모델 ensemble `성공`
- Mixup `실패`

## Link
- [실험 과정 공유](https://www.notion.so/3365343dd8474b259141ce730e1afe0f?v=cb0e96e64fda4c4a9d8d97914b8234bf)
- [실험 결과 공유](https://docs.google.com/spreadsheets/d/12upH-lAlvF2PtLd0D70Nqz_DThTubp86_443BF-xmZE/edit?usp=sharing)
