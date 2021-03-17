"""
This python script converts the network into Script Module
"""
import torch
import numpy as np
import os
import cv2

import net_architectures.GRDN as net
import utils.utils as utils
import torchvision.transforms as transforms


def recon(batch_tensor, net, scale_factor, odd):
    # network 에 downscaling 부분이 있으면 영상 사이즈가 downscaling 하는 수 만큼 영상에 padding 을 해줘야 한다.
    # padding 이 된 영상이 network 를 통과한 후 padding 을 지워준다.
    pad = utils.TorchPaddingForOdd(odd, scale_factor=scale_factor)
    batch_tensor = pad.padding(batch_tensor)
    device = torch.device(f'cuda:0')

    with torch.no_grad():
        net.eval()

        # 속도 측정
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        batch_tensor = batch_tensor.to(device)
        start.record()
        batch_tensor_out = net(batch_tensor)
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        print(f'time : {start.elapsed_time(end)}')

        batch_tensor_out = pad.unpadding(batch_tensor_out)

        batch_tensor_out = batch_tensor_out.cpu()
        return batch_tensor_out


def recon_one_channel_frame(img_ori, net, scale_factor, downupcount):
    # 한 frame 을 복원하는 함수이다.

    # input_ 를 tensor 로 전환해준다.
    totensor = transforms.ToTensor()  # [0,255] -> [0,1] 로 만들어줌.
    input_img = totensor(img_ori)

    # conv2d 를 하려면 input 이 4 channel 이어야 한다!
    input_img = input_img.view(1, -1, input_img.shape[1], input_img.shape[2])

    # 복원하기.
    output_img = recon(input_img, net, scale_factor, downupcount)

    # 0,1 -> 0,255
    npimg = output_img.numpy() * 255

    # 영상 후처리.
    npimg = np.around(npimg)
    npimg = npimg.clip(0, 255)
    npimg = npimg.astype(np.uint8)
    # tensor 를 pil 또는 cv2 형태로 바꾸기 위해 channel 순서를 바꿔줌 (batch, channel, w, h) -> (batch, w, h, channel)
    npimg = np.transpose(npimg, (0, 2, 3, 1))
    npimg = np.squeeze(npimg)

    return npimg


if __name__ == '__main__':

    # >>> 사용할 gpu 번호.
    cuda_num = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    print('\n===> cuda_num :', cuda_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num
    device = torch.device(f'cuda:0')

    # >>> 사용할 딥러닝 모델들을 불러온다.
    net_dict = {'G': net.GRDN(input_channel=1)}

    # >>> 불러온 모델의 사이즈를 출력해본다.
    print(f'\n===> Model size')
    for key in net_dict.keys():
        print(f'Number of params ({key}): '
              f'{sum([p.data.nelement() for p in net_dict[key].parameters()])}')

    # >>> 모델을 GPU 로 보내준다.
    netG = net_dict['G'].to(device)

    # >>> 저장된 모델을 불러온다.
    def weight_loader(load_checkpoint_dir):
        print(f"\n===> Load checkpoint")
        if os.path.isfile(load_checkpoint_dir):
            print(": loading checkpoint '{}'".format(load_checkpoint_dir))
            checkpoint = torch.load(load_checkpoint_dir)
            netG.load_state_dict(checkpoint['G'])
            iter_count = checkpoint['iter_count']
            print('iter_count :', iter_count)
        else:
            print(": no checkpoint found at '{}'".format(load_checkpoint_dir))
            return 0, 0  # iter_count, best_psnr
    # 모델이 저장된 dir.
    load_checkpoint_dir = "resources/hevc009_RA_qp37/checkpoint_psnr_best_37-average.pth"
    weight_loader(load_checkpoint_dir)

    # Set upgrading the gradients to False
    for param in netG.parameters():
        param.requires_grad = False

    # >>> yuv420 영상을 읽어본다.
    input_img_dir = "resources/C01_BasketballDrill_832x480_50_QP37.yuv"
    target_img_dir = "resources/C01_BasketballDrill_832x480_50.yuv"
    w = 832
    h = 480
    target_POC = 0
    channel = 'y_u_v'
    input_y_u_v, w_input, h_input = utils.read_one_from_yuvs(input_img_dir, w, h, target_POC, channel=channel)
    target_y_u_v, w_input, h_input = utils.read_one_from_yuvs(target_img_dir, w, h, target_POC, channel=channel)

    # >>> 영상을 복원한다.
    y_recon = recon_one_channel_frame(input_y_u_v['y'], netG, scale_factor=1, downupcount=2)  # 복원 영상이 np 이다.
    y_recon = recon_one_channel_frame(input_y_u_v['y'], netG, scale_factor=1, downupcount=2)  # 복원 영상이 np 이다.
    y_recon = recon_one_channel_frame(input_y_u_v['y'], netG, scale_factor=1, downupcount=2)  # 복원 영상이 np 이다.
    y_recon = recon_one_channel_frame(input_y_u_v['y'], netG, scale_factor=1, downupcount=2)  # 복원 영상이 np 이다.
    y_recon = recon_one_channel_frame(input_y_u_v['y'], netG, scale_factor=1, downupcount=2)  # 복원 영상이 np 이다.
    u_recon = recon_one_channel_frame(input_y_u_v['u'], netG, scale_factor=1, downupcount=2)
    v_recon = recon_one_channel_frame(input_y_u_v['v'], netG, scale_factor=1, downupcount=2)
    y_u_v_recon_merged = np.hstack([y_recon.flatten(), u_recon.flatten(), v_recon.flatten()])

    # >>> save png
    print('Saving image... (This may take some time.)')
    yuv_reshaped = np.reshape(y_u_v_recon_merged, [int(h * 1.5), w])
    bgr = cv2.cvtColor(yuv_reshaped, cv2.COLOR_YUV2BGR_I420)
    cv2.imwrite(f'resources/recon_{str(target_POC).zfill(3)}.png', bgr)

    # >>> get PSNR
    psnr_y = utils.get_psnr(y_recon.astype(np.float32), target_y_u_v['y'].astype(np.float32), 0, 255)
    print(f"psnr : {psnr_y}")

    ################################################################################################
    # >>> ScriptModule 로 변환
    print(f"\n===> to ScriptModule")
    x = torch.rand(1, 1, 240, 416).to(device)
    netG_traced = torch.jit.trace(netG, x)
    netG_traced_dir = 'resources/hevc009_RA_qp37/netG_traced.pt'
    netG_traced.save(netG_traced_dir)
    print(f"..")

    y_recon = recon_one_channel_frame(input_y_u_v['y'], netG_traced, scale_factor=1, downupcount=2)  # 복원 영상이 np 이다.
    y_recon = recon_one_channel_frame(input_y_u_v['y'], netG_traced, scale_factor=1, downupcount=2)  # 복원 영상이 np 이다.
    y_recon = recon_one_channel_frame(input_y_u_v['y'], netG_traced, scale_factor=1, downupcount=2)  # 복원 영상이 np 이다.
    y_recon = recon_one_channel_frame(input_y_u_v['y'], netG_traced, scale_factor=1, downupcount=2)  # 복원 영상이 np 이다.
    y_recon = recon_one_channel_frame(input_y_u_v['y'], netG_traced, scale_factor=1, downupcount=2)  # 복원 영상이 np 이다.

    u_recon = recon_one_channel_frame(input_y_u_v['u'], netG_traced, scale_factor=1, downupcount=2)
    v_recon = recon_one_channel_frame(input_y_u_v['v'], netG_traced, scale_factor=1, downupcount=2)
    y_u_v_recon_merged = np.hstack([y_recon.flatten(), u_recon.flatten(), v_recon.flatten()])

    # >>> save png
    print('Saving image... (This may take some time.)')
    yuv_reshaped = np.reshape(y_u_v_recon_merged, [int(h * 1.5), w])
    bgr = cv2.cvtColor(yuv_reshaped, cv2.COLOR_YUV2BGR_I420)
    cv2.imwrite(f'resources/recon_traced_{str(target_POC).zfill(3)}.png', bgr)

    # >>> get PSNR
    psnr_y = utils.get_psnr(y_recon.astype(np.float32), target_y_u_v['y'].astype(np.float32), 0, 255)
    print(f"psnr (traced): {psnr_y}")


    print(netG_traced.graph)

    ######################################################################################################
    # >>> 저장된 ScriptModule 을 가져와서 추론해본다.
    print(f"\n===> load ScriptModule")
    netG_traced_loaded = torch.jit.load(netG_traced_dir)
    print(f"..")

    print(input_y_u_v['y'].shape)
    y_recon = recon_one_channel_frame(input_y_u_v['y'], netG_traced_loaded, scale_factor=1, downupcount=2)  # 복원 영상이 np 이다.
    y_recon = recon_one_channel_frame(input_y_u_v['y'], netG_traced_loaded, scale_factor=1, downupcount=2)  # 복원 영상이 np 이다.
    y_recon = recon_one_channel_frame(input_y_u_v['y'], netG_traced_loaded, scale_factor=1, downupcount=2)  # 복원 영상이 np 이다.
    y_recon = recon_one_channel_frame(input_y_u_v['y'], netG_traced_loaded, scale_factor=1, downupcount=2)  # 복원 영상이 np 이다.
    y_recon = recon_one_channel_frame(input_y_u_v['y'], netG_traced_loaded, scale_factor=1, downupcount=2)  # 복원 영상이 np 이다.
    u_recon = recon_one_channel_frame(input_y_u_v['u'], netG_traced_loaded, scale_factor=1, downupcount=2)
    v_recon = recon_one_channel_frame(input_y_u_v['v'], netG_traced_loaded, scale_factor=1, downupcount=2)
    y_u_v_recon_merged = np.hstack([y_recon.flatten(), u_recon.flatten(), v_recon.flatten()])

    # >>> save png
    print('Saving image... (This may take some time.)')
    yuv_reshaped = np.reshape(y_u_v_recon_merged, [int(h * 1.5), w])
    bgr = cv2.cvtColor(yuv_reshaped, cv2.COLOR_YUV2BGR_I420)
    cv2.imwrite(f'resources/recon_traced_loaded_{str(target_POC).zfill(3)}.png', bgr)

    # >>> get PSNR
    psnr_y = utils.get_psnr(y_recon.astype(np.float32), target_y_u_v['y'].astype(np.float32), 0, 255)
    print(f"psnr (traced): {psnr_y}")




