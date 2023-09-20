python Try_on_ICPN.py --type 1 --warp_checkpoint ./checkpoints/Pruning/WarpModel_ICPN_0.5.pth > output/WarpModel_ICPN_0.5.txt
python Try_on_QAT.py --type 1 --warp_checkpoint ./checkpoints/Pruning/WarpModel_ICPN_0.5.pth > output/WarpModel_QAT_Pytorch.txt
python Try_on_Gen.py --name Try_on_Gen_0.2 --warp_checkpoint ./checkpoints/Pruning/WarpModel_Pruned_0.2.pth --gen_checkpoint ./checkpoints/Pruning/GenModel_pruned_0.2.pth > output/GenModel_Pruned_0.2.txt

# neural compressor
python Try_on_Warp.py --warp_checkpoint ./checkpoints/Pruning/WarpModel_ICPN_0.5.pth > output/WarpModel_NC.txt
python Try_on_model.py --name Try_on_model
python OpenvinoModel.py --name Openvino_model