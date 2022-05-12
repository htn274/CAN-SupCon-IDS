# CAN-SupCon-IDS

## How to run

### Baseline train

```
python3 train_baseline.py --data_dir ../Data/TFrecord_w29_s15/ --model resnet18 --save_freq 10 --window_size 29 --num_workers 8 --cosine --epochs 50 --batch_size 256 --learning_rate 0.0005 --rid 5
```

### Supcon train

```
python3 train_supcon.py --data_dir ../Data/TFrecord_w29_s15/ --model resnet18 --save_freq 10 --window_size 29 --epochs 200 --num_workers 8 --temp 0.07 --learning_rate 0.1 --learning_rate_classifier 0.01 --cosine --epoch_start_classifier 170 --rid 3 --batch_size 512
```

### Transfer

```
python3 transfer.py --data_path ../Data/Survival/ --car_model Spark --window_size 29 --num_classes 4 --lr_transfer 0.01 --lr_tune 0.001 --transfer_epochs 50 --tune_epochs 10 --tf_algo transfer_tune --pretrained_model resnet --pretrained_path save/resnet181_lr0.0005_bs256_50epochs_050122_093024_cosine/models/ --source_ckpt 50
```

```
python3 transfer.py --data_path ../Data/Survival/ --car_model Spark --window_size 29 --num_classes 4 --lr_transfer 0.01 --lr_tune 0.001 --transfer_epochs 40 --tune_epochs 20 --tf_algo transfer_tune --pretrained_model supcon --pretrained_path ./save/SupCon_resnet182_lr0.1_0.01_bs1024_200epoch_temp0.07_042822_144940_cosine_warm/models/ --source_ckpt 100
```

### Train test split

```
python3 train_test_split.py --data_path ../Data/ --window_size 29 --strided 15 --rid 2
```