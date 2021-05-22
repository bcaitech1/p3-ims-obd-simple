_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='VFNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=11,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# data setting
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
        type='VerticalFlip',
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[-0.1, 0.1],
        contrast_limit=[-0.1, 0.1],
        p=0.7),
    dict(
        type='CLAHE',
        clip_limit=2.0),
    dict(
        type='HueSaturationValue',
        hue_shift_limit=10,
        sat_shift_limit=15,
        val_shift_limit=10),
    dict(
        type='RandomRotate90'),
    dict(
        type='OneOf',
        transforms=[
            dict(type='MedianBlur', blur_limit=5),
            dict(type='MotionBlur', blur_limit=5),
            dict(type='GaussianBlur', blur_limit=5),
        ],
        p=0.7),
    dict(
        type='GaussNoise', var_limit=(5.0, 30.0), p=0.5)
]


train_pipeline = [
    dict(type='LoadImageFromFile'),
#      dict(
#         type='InstaBoost',
#         action_candidate=('normal', 'horizontal', 'skip'),
#         action_prob=(1, 0, 0),
#         scale=(0.8, 1.2),
#         dx=15,
#         dy=15,
#         theta=(-1, 1),
#         color_prob=0.,
#         hflag=False,
#         aug_ratio=0.5),
    
    dict(type='LoadAnnotations', with_bbox=True),
     dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(512, 512),(576, 576),(640, 640),(704, 704),(768, 768),
                      (832, 832),(896, 896),(960, 960),(1024, 1024)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(512, 512), (768, 768), (1024, 1024)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(512, 512),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(512, 512),(576, 576),(640, 640),(704, 704),(768, 768),
                      (832, 832),(896, 896),(960, 960),(1024, 1024)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
#     dict(type='AutoAugment',
#          policies=[
#              [
#                  dict(type='Mosaic', mosaic_ratio=0.5),  ## 추가한 것
#              ]
#          ]),
    
#     dict(type='Mosaic', mosaic_ratio=0.5), 
#     dict(type='MyTransform' ,mosaic_ratio=0.5),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),x
#     dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
     dict( 
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='SegRescale', scale_factor=1 / 8),
    dict(type='DefaultFormatBundle'),
    
    
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))


optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))


fp16 = None

# lr_config = dict(policy='CosineAnnealing',warmup='linear',warmup_iters=3000,
#                     warmup_ratio=0.0001, min_lr_ratio=1e-7)
runner = dict(type='EpochBasedRunnerAmp', max_epochs=60)

# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )


# optimizer
optimizer = dict(
    lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))

optimizer_config = dict(grad_clip=None)
# # learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)
