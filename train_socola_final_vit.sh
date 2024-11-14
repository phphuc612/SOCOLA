CUDA_VISIBLE_DEVICES=4 python main_socola_final.py --arch vit_b_16 \
                    --seed 42 \
                    --lr 0.005 \
                    --batch-size 1024 \
                    --workers 2 \
                    --epochs 50 \
                    --cos \
                    --mlp \
                    --aug-plus \
                    --subset 1 \
                    --print-freq 5 \
                    --save-dir "ckpts-socola-final-vit" \
                    --run-name "socola-final-vit" \
                    "/home/ubuntu/work/data"