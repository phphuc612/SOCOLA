CUDA_VISIBLE_DEVICES=6 python main_socola_final_no_temp.py --arch resnet50 \
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
                    --save-dir "ckpts-socola-final-no-temp-rn50" \
                    --run-name "socola-final-no-temp-resnet50" \
                    --resume "/home/ubuntu/work/SOCOLA/ckpts-socola-final-no-temp-rn50/checkpoint_0004.pth.tar" \
                    "/home/ubuntu/work/data"