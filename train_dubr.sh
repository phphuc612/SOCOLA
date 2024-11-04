python main_dual_branch.py \
    --arch resnet50 \
    --lr 0.005 \
    --batch-size 32 \
    --sub-batch-size 2 \
    --T 0.07 \
    --workers 2 \
    --epochs 1 \
    --cos \
    --mlp \
    --aug-plus \
    --subset 1 \
    --save-dir "dual_branch-ckpts-shuffle" \
    "/mnt/nvme0n1p2/CVPR" 
    # --resume "ckpts-shuffle/checkpoint_0025.pth.tar" \

