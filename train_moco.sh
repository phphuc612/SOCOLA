python main_moco.py --arch resnet50 \
                    --lr 0.005 \
                    --batch-size 128 \
                    --workers 2 \
                    --epochs 30 \
                    --cos \
                    --mlp \
                    --aug-plus \
                    --subset 1 \
                    --save-dir "ckpts-moco-rn50" \
                    --run-name "moco-resnet50" \
                    "/mnt/nvme0n1p2/CVPR"