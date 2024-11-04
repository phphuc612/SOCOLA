python main_dual_branch.py  --arch resnet50 \
                            --lr 0.005 \
                            --batch-size 32 \
                            --sub-batch-size 8 \
                            --T 0.07 \
                            --workers 2 \
                            --epochs 3 \
                            --cos \
                            --mlp \
                            --aug-plus \
                            --subset 1 \
                            --save-dir "ckpts-test" \
                            "/mnt/nvme0n1p2/CVPR"
python main_dual_branch.py  --arch resnet50 \
                            --lr 0.005 \
                            --batch-size 128 \
                            --sub-batch-size 8 \
                            --T 0.07 \
                            --workers 2 \
                            --epochs 3 \
                            --cos \
                            --mlp \
                            --aug-plus \
                            --subset 1 \
                            --save-dir "ckpts-test" \
                            "/mnt/nvme0n1p2/CVPR"
python main_dual_branch.py  --arch resnet50 \
                            --lr 0.005 \
                            --batch-size 512 \
                            --sub-batch-size 8 \
                            --T 0.07 \
                            --workers 2 \
                            --epochs 3 \
                            --cos \
                            --mlp \
                            --aug-plus \
                            --subset 1 \
                            --save-dir "ckpts-test" \
                            "/mnt/nvme0n1p2/CVPR"
python main_dual_branch.py  --arch resnet50 \
                            --lr 0.005 \
                            --batch-size 1024 \
                            --sub-batch-size 8 \
                            --T 0.07 \
                            --workers 2 \
                            --epochs 3 \
                            --cos \
                            --mlp \
                            --aug-plus \
                            --subset 1 \
                            --save-dir "ckpts-test" \
                            "/mnt/nvme0n1p2/CVPR"



