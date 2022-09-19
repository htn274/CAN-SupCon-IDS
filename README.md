# CAN-SupCon-IDS

This is the implementation of the paper ["SupCon ResNet and Transfer Learning for the In-vehicle Intrusion Detection System"](https://arxiv.org/submit/4407238/view)

## Environment 

- tensorflow: 2.0
- torch: 1.9

## How to run

### Train test split

```
python3 train_test_split.py --data_path ../Data/ --car_model None --window_size 29 --strided 15 --rid 2
```

### Baseline train

```
python3 train_baseline.py --data_dir ../Data/TFrecord_w29_s15/ \\
                          --model resnet18 --save_freq 10 --window_size 29 \\
                          --num_workers 8 --cosine --epochs 50 \\
                          --batch_size 256 --learning_rate 0.0005 --rid 5
```

### Supcon train

```
python3 train_supcon.py --data_dir ../Data/TFrecord_w29_s15/ \\
                      --model resnet18 --save_freq 10 --window_size 29 \\
                      --epochs 200 --num_workers 8 --temp 0.07 \\
                      --learning_rate 0.1 --learning_rate_classifier 0.01 \\
                      --cosine --epoch_start_classifier 170 --rid 3 --batch_size 512
```

### Transfer

Random initialization

```
python3 transfer.py --data_path ../Data/Survival/ --car_model Spark \\
                    --pretrained_model resnet --tf_algo tune \\
                    --num_classes 4 --window_size 29 --strided 10 \\
                    --lr_tune 0.001 --tune_epochs 20
```

Using CE ResNet as the pretrained model

```
python3 transfer.py --data_path ../Data/Survival/ --car_model Spark \\
                    --window_size 29 --strided 10 --num_classes 4 --lr_transfer 0.01 \\
                    --lr_tune 0.001 --transfer_epochs 50 --tune_epochs 10 \\
                    --tf_algo transfer_tune --pretrained_model resnet \\
                    --pretrained_path save/smallresnet18.ce1_gamma0_lr0.001_bs256_50epochs_051822_100142_cosine/models/ \\
                    --source_ckpt 50
```

Using SupCon ResNet as the pretrained model

```
python3 transfer.py --data_path ../Data/Survival/ --car_model Spark \\
                    --window_size 29 --strided 10 --num_classes 4 --lr_transfer 0.01 \\
                    --lr_tune 0.001 --transfer_epochs 40 --tune_epochs 20 \\
                    --tf_algo transfer_tune --pretrained_model supcon \\
                    --pretrained_path save/SupCon_resnet18.ce2_lr0.05_0.01_bs512_200epoch_temp0.07_052322_102305_cosine_warm/models/ \\
                    --source_ckpt 200
```

## Acknowledgement

This codebase was adapted from [SupContrast](https://github.com/HobbitLong/SupContrast).
