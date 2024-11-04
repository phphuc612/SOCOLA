python resnet.py --arch resnet50 \
                    --batch-size 32 \
                    --workers 2 \
                    --epochs 30 \
                    --subset 1 \
                    --lr 0.00001 \
                    --pretrained "/mnt/nvme0n1p2/Moco-MAC/SOCOLAClone/ckpts/checkpoint_0029.pth.tar" \
                    "/mnt/nvme0n1p2/CVPR"


