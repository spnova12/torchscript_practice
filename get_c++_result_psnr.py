"""
This python script converts the network into Script Module
"""
import torch
import numpy as np
import os
import cv2
import python_kdw.utils.utils as utils


if __name__ == '__main__':

    # >>> yuv420 영상을 읽어본다.
    input_c_dir = "python_kdw/resources/C01_BasketballDrill_832x480_50_QP37_c++_recon.yuv"
    input_python_dir = "python_kdw/resources/recon_traced_loaded_000.yuv"

    gt_img_dir = "python_kdw/resources/C01_BasketballDrill_832x480_50.yuv"

    w = 832
    h = 480
    target_POC = 0
    channel = 'y_u_v'

    input_c_y_u_v, w_input, h_input = utils.read_one_from_yuvs(input_c_dir, w, h, target_POC, channel=channel)
    input_python_y_u_v, _, _ = utils.read_one_from_yuvs(input_python_dir, w, h, target_POC, channel=channel)
    target_y_u_v, _, _ = utils.read_one_from_yuvs(gt_img_dir, w, h, target_POC, channel=channel)

    # >>> get PSNR
    input_c_psnr = utils.get_psnr(input_c_y_u_v['y'].astype(np.float32), target_y_u_v['y'].astype(np.float32), 0, 255)
    print(f"input_c_psnr : {input_c_psnr}")

    input_python_psnr = utils.get_psnr(input_python_y_u_v['y'].astype(np.float32), target_y_u_v['y'].astype(np.float32), 0, 255)
    print(f"input_python_psnr : {input_python_psnr}")
