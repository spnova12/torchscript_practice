"""
This python script converts the network into Script Module
"""
import torch
import os

import net_architectures.GRDN as net
import utils.utils as utils
import torchvision.transforms as transforms

# yuv420 영상을 읽어본다.
input_img_dir = "resources/C01_BasketballDrill_832x480_50_QP37.yuv"
target_img_dir = "resources/C01_BasketballDrill_832x480_50.yuv"
w = 832
h = 480

start_frame = 0
channel = 'y'

input_img, w_input, h_input = utils.read_one_from_yuvs(input_img_dir, w, h, start_frame, channel=channel)
target_img, w_input, h_input = utils.read_one_from_yuvs(target_img_dir, w, h, start_frame, channel=channel)

# tensor로 바꿔주기.
totensor = transforms.ToTensor()  # [0,255] -> [0,1] 로 만들어줌.
input_img = totensor(input_img)
target_img = totensor(target_img)

print(input_img)







#
#
# # 모델을 만든다.
# netG = net.GRDN(input_channel=1)
# load_checkpoint_dir = "hevc009_RA_qp37/checkpoint_psnr_best_37-average.pth"
#
# if os.path.isfile(load_checkpoint_dir):
#     print(": loading checkpoint '{}'".format(load_checkpoint_dir))
#     checkpoint = torch.load(load_checkpoint_dir)
#     iter_count = checkpoint['iter_count']
#     best_psnr = checkpoint['best_psnr']
#     netG.load_state_dict(checkpoint['G'])
# else:
#     print(": no checkpoint found at '{}'".format(load_checkpoint_dir))
#
#
#
#
#
#
#
# # Set upgrading the gradients to False
# for param in netG.parameters():
#     param.requires_grad = False
#
# example_input = torch.rand(1, 1, 240, 416)
# script_module = torch.jit.trace(netG, example_input)
# script_module.save('hevc008_RA_qp37_fit_checkpoint_psnr_best_37-average.pt')
