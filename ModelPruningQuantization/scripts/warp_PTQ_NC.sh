python warp_PTQ_NC.py --name warp_ptq_NC \
--PBAFN_warp_checkpoint 'checkpoints/PBAFN_e2e/PBAFN_warp_epoch_101.pth' --PBAFN_gen_checkpoint 'checkpoints/PBAFN_e2e/PBAFN_gen_epoch_101.pth' \
--PFAFN_warp_checkpoint 'checkpoints/pruning/WarpModel_ICPN_0.5.pth' \
--batchSize 45 --label_nc 14 > output/warp_ptq_NC.txt