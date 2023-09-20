python warp_pruning_finetuning_ICPN.py --name warp_pruning_finetuning_ICPN  \
--PBAFN_warp_checkpoint 'checkpoints/PBAFN_e2e/PBAFN_warp_epoch_101.pth' --PBAFN_gen_checkpoint 'checkpoints/PBAFN_e2e/PBAFN_gen_epoch_101.pth'  \
--PFAFN_warp_checkpoint 'checkpoints/PFAFN_warp_epoch_101.pth' \
--lr 0.00001 --niter 10 --niter_decay 10 --batchSize 45 --label_nc 14 > output/warp_pruning_finetuning_ICPN.txt










