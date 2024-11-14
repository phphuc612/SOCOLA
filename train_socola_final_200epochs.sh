CUDA_VISIBLE_DEVICES=0 python main_socola_final.py --arch resnet50 \
                    --seed 42 \
                    --lr 0.005 \
                    --batch-size 1024 \
                    --workers 2 \
                    --epochs 200 \
                    --cos \
                    --mlp \
                    --aug-plus \
                    --subset 1 \
                    --print-freq 5 \
                    --save-dir "ckpts-socola-final-rn50-200epochs" \
                    --run-name "socola-final-resnet50-200epochs" \
                    "/home/ubuntu/work/data"