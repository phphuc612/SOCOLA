for i in 29
do
python main_moco.py --arch resnet50 \
                    --lr 0.005 \
                    --batch-size 1024 \
                    --sub-batch-size 8 \
                    --T 0.07 \
                    --workers 2 \
                    --epochs 30 \
                    --cos \
                    --mlp \
                    --aug-plus \
                    --subset 1 \
                    --eval-only "/mnt/nvme0n1p2/Moco-MAC/SOCOLAClone/ckpts-shuffle/checkpoint_$(printf "%04d" $i).pth.tar" \
                    "/mnt/nvme0n1p2/CVPR"
done