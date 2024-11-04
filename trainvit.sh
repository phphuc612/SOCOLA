python main_moco.py --arch vit_b_16 \
                    --lr 0.005 \
                    --batch-size 2048 \
                    --sub-batch-size 8 \
                    --T 0.07 \
                    --workers 2 \
                    --epochs 30 \
                    --cos \
                    --aug-plus \
                    --subset 1 \
                    --run-name "shuffle-vit-2" \
                    --save-dir "ckpts-shuffle-vit2" \
                    "/mnt/nvme0n1p2/CVPR"
                    # --resume "ckpts-shuffle-vit/checkpoint_0004.pth.tar" \

