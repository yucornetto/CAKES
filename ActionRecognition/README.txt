Due to the size limitation, we mainly provide the code for video dataset, and pre-trained weight for CAKES123D-Performance-Priority trained with 8-frame/16-frame settings respectively. We provide the command to reproduce the experiments as follows:

We follow "Temporal Relational Reasoning in Videos, Zhou et.al, ECCV2018" for the dataset preparation.

The code is tested under enviroment with TITAN RTX, CUDA 10.1, PyTorch 1.0

### Search for CAKES123D, CAKES12D, CAKES23D Performance Priority
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --root_path PATH_TO_DATASET --dataset somethingv1 --checkpoint_dir PATH_TO_SAVE_CKPT --type I3D --arch resnet50 --num_segments 8 -b 96 --lr 0.04 --search --op_code conv1_2_3d_dropout --epochs 140 --lr_steps 100 120
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --root_path PATH_TO_DATASET --dataset somethingv1 --checkpoint_dir PATH_TO_SAVE_CKPT --type I3D --arch resnet50 --num_segments 8 -b 96 --lr 0.04 --search --op_code convst_dropout --epochs 140 --lr_steps 100 120
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --root_path PATH_TO_DATASET --dataset somethingv1 --checkpoint_dir PATH_TO_SAVE_CKPT --type I3D --arch resnet50 --num_segments 8 -b 96 --lr 0.04 --search --op_code conv1_2d_dropout --epochs 140 --lr_steps 100 120


### Search for CAKES123D Cost Priority
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --root_path PATH_TO_DATASET --dataset somethingv1 --checkpoint_dir PATH_TO_SAVE_CKPT --type I3D --arch resnet50 --num_segments 8 -b 96 --lr 0.04 --op_code conv1_2_3d -sr --s 0.0001 --search --reweight
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --root_path PATH_TO_DATASET --dataset somethingv1 --checkpoint_dir PATH_TO_SAVE_CKPT --type I3D --arch resnet50 --num_segments 8 -b 96 --lr 0.04 --op_code convst -sr --s 0.0001 --search --reweight
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --root_path PATH_TO_DATASET --dataset somethingv1 --checkpoint_dir PATH_TO_SAVE_CKPT --type I3D --arch resnet50 --num_segments 8 -b 96 --lr 0.04 --op_code conv1_2d -sr --s 0.0001 --search --reweight


##### To train and evaluate the found CAKES (different CAKES conv configuration can be found in config.py)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --root_path PATH_TO_DATASET --dataset somethingv1 --checkpoint_dir PATH_TO_SAVE_CKPT --type I3D --arch resnet50 --num_segments 8 -b 96 --lr 0.04 --op_code conv1_2_3d --conv_config CAKES123D_P

#### To test with the provided weight

### 8frame
### Testing Results: Prec@1 47.240 Prec@5 75.673 Loss 2.39457
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --root_path PATH_TO_DATASET --dataset somethingv1 --checkpoint_dir PATH_TO_SAVE_CKPT --type I3D --arch resnet50 --num_segments 8 -b 96 --lr 0.04 --op_code conv1_2_3d --conv_config CAKES123D_P --resume ./ckpt/8frame_CAKES123D_PerformancePriority.tar --evaluate

### 16frame
### Testing Results: Prec@1 49.410 Prec@5 78.363 Loss 2.27473
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --root_path PATH_TO_DATASET --dataset somethingv1 --checkpoint_dir PATH_TO_SAVE_CKPT --type I3D --arch resnet50 --num_segments 16 -b 96 --lr 0.04 --op_code conv1_2_3d --conv_config CAKES123D_P --resume ./ckpt/16frame_CAKES123D_PerformancePriority.tar --evaluate
