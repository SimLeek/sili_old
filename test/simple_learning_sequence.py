from sili.modules.image_pyramid.image_pyramid import ToImagePyramid
from sili.modules.depth_pyr_conv.depth_pyr_conv import DepthPyrConv, depth_edge_matrix
from sili.modules.mse_loss.mse_loss import MSELoss
from sili.core.devices.gpu import GPUManager

import cv2
import numpy as np

if __name__ == "__main__":
    im = cv2.imread("../../../test/files/test_ai_pls_ignore.png").astype(np.float32) / 255

    gpu = GPUManager()

    im_pyr = ToImagePyramid(gpu, im)
    depth_pyr_conv = DepthPyrConv(gpu, im_pyr.out_pyr, depth_edge_matrix)

    run_list = [
        im_pyr,
        depth_pyr_conv
    ]

    runner.run_once(run_list)