from argparse import ArgumentParser


class BaseOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.opt = None
        self.parser.add_argument('--norm', type=str, default='instance', help='')
        self.parser.add_argument('--name', type=str, default='instance', help='work space name')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')

        self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')

        self.parser.add_argument('--QAT_checkpoint', type=str, default='quantized/gen_QAT_Pytorch.pth',
                                 help='load the quantized model from the specified location')
        self.parser.add_argument('--warp_checkpoint', type=str, default='PFAFN_warp_epoch_101.pth',
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--gen_checkpoint', type=str, default='PFAFN_gen_epoch_101.pth',
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--type', type=int, default=0, help='choose a pruned module')
        self.parser.add_argument('--save_path', type=str, default='result', help='save the try_on image')
        self.parser.add_argument('--image_path', type=str, default='dataset/images_origin', help='images path')
        self.parser.add_argument('--clothe_path', type=str, default='dataset/clothes_origin', help='clothes path')
        self.parser.add_argument('--label_path', type=str, default='dataset/train_label', help='labels path')
        self.parser.add_argument('--save_checkpoint', type=str, default='./checkpoints/Pruning',
                                 help='save the prunned model')
    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
