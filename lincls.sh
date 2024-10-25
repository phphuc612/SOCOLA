python main_lincls.py --arch resnet50 \
                    --batch-size 256 \
                    --workers 8 \
                    --epochs 30 \
                    --subset 1 \
                    --lr 0.0001 \
                    --save-dir "linearcls" \
                    --pretrained "/mnt/nvme0n1p2/Moco-MAC/SOCOLAClone/ckpts/checkpoint_0029.pth.tar" \
                    "/mnt/nvme0n1p2/CVPR"


