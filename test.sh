python main_moco.py --arch resnet50 \
                    --lr 0.005 \
                    --batch-size 4096 \
                    --sub-batch-size 8 \
                    --T 0.07 \
                    --workers 2 \
                    --epochs 30 \
                    --cos \
                    --mlp \
                    --aug-plus \
                    --subset 1 \
                    --save-dir "ckpts-test" \
                    "/mnt/nvme0n1p2/CVPR"

