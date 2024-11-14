CUDA_VISIBLE_DEVICES=1 python main_moco.py --arch resnet50 \
                    --seed 42 \
                    --lr 0.005 \
                    --batch-size 256 \
                    --workers 2 \
                    --epochs 50 \
                    --cos \
                    --mlp \
                    --aug-plus \
                    --subset 1 \
                    --print-freq 5 \
                    --save-dir "ckpts-moco-rn50" \
                    --run-name "moco-resnet50" \
                    "/home/ubuntu/work/data"