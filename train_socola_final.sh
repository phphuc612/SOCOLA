python main_socola_final.py --arch resnet50 \
                    --seed 42 \
                    --lr 0.005 \
                    --batch-size 1024 \
                    --workers 2 \
                    --epochs 50 \
                    --cos \
                    --mlp \
                    --aug-plus \
                    --subset 0.01 \
                    --print-freq 10 \
                    --save-dir "ckpts-socola-final-rn50" \
                    --run-name "socola-final-resnet50" \
                    "/mnt/nvme0n1p2/CVPR"