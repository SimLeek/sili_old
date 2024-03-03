from sili.modules.image_pyramid.image_pyramid import ToImagePyramid
from sili.modules.depth_pyr_conv.depth_pyr_conv import DepthPyrConv, depth_edge_matrix
from sili.modules.mse_loss.mse_loss import MSELoss
from sili.modules.adacon.adacon import Adacon
from sili.core.devices.gpu import GPUManager
from displayarray import DirectDisplay
import cv2
import numpy as np
import kp

class GPUMLPipeline(object):
    def __init__(self, gpu, module_list, input_buffers=None, output_buffers=None):
        self.gpu = gpu
        self.module_list = module_list
        self.input_buffers = input_buffers
        self.output_buffers = output_buffers

        self._setup_pipeline()

    def _setup_pipeline(self):
        self.upload_sequence = self.gpu.manager.sequence()
        for m in self.module_list:
            if m.has_forward and m.forward_input_buffers:
                self.upload_sequence.record(kp.OpTensorSyncDevice([*m.forward_input_buffers]))
        for m in reversed(self.module_list):
            if m.has_backward and m.backward_input_buffers:
                self.upload_sequence.record(kp.OpTensorSyncDevice([*m.backward_input_buffers]))
        for m in self.module_list:
            if m.has_optim and m.optim_input_buffers:
                self.upload_sequence.record(kp.OpTensorSyncDevice([*m.optim_input_buffers]))
        self.upload_sequence.eval()

        self.full_sequence = self.gpu.manager.sequence()
        self.full_sequence.record(kp.OpTensorSyncDevice([*self.input_buffers]))
        for m in self.module_list:
            for f in m.forward_ops():
                self.full_sequence.record(f)
        for m in reversed(self.module_list):
            for b in m.backward_ops():
                self.full_sequence.record(b)
        for m in self.module_list:
            for opt in m.optim_ops():
                self.full_sequence.record(opt)
        self.full_sequence.record(kp.OpTensorSyncLocal([*self.output_buffers]))


    def run_once(self, *args):
        for ip_buf, arg in zip(self.input_buffers, args):
            ip_buf.data()[:] = arg.flat
        self.full_sequence.eval()
        return self.output_buffers



if __name__ == "__main__":
    im = cv2.imread("files/test_ai_pls_ignore.png").astype(np.float32) / 255

    gpu = GPUManager()

    # err: grad and contrib seem to both be zero

    im_pyr = ToImagePyramid(gpu, im)
    depth_pyr_conv = DepthPyrConv(gpu, im_pyr.out_pyr, depth_edge_matrix, backprop_conv=True)
    depth_pyr_conv_optim = Adacon(gpu,
                                  depth_pyr_conv.depth_conv.buffer,
                                  depth_pyr_conv.depth_conv_err.buffer,
                                  depth_pyr_conv.depth_conv_contrib.buffer,
                                  depth_pyr_conv.depth_conv_str.buffer,
                                  )
    mse_loss = MSELoss(gpu,
                       im_pyr.out_pyr.image_buffer,
                       depth_pyr_conv.out_pyr.image_buffer,
                       depth_pyr_conv.out_pyr_err.image_buffer)

    # will be run in a zigzag pattern:
    #   forward:  top to bottom
    #   backward: bottom to top
    #   optim:    top to bottom
    runner = GPUMLPipeline(
        gpu,
        [
            im_pyr,
            depth_pyr_conv,
            depth_pyr_conv_optim,
            mse_loss
        ],
        # set these so that only these are moved to/from the gpu
        # can be set to error or other info for debugging
        input_buffers=im_pyr.forward_input_buffers,
#        output_buffers=depth_pyr_conv.forward_output_buffers
        output_buffers=depth_pyr_conv_optim.optim_output_buffers + [depth_pyr_conv.buf_conv_prepool] + [depth_pyr_conv.out_pyr_err.image_buffer]
    )
    d = DirectDisplay()
    while not d.window.is_closing:
        out_buf = runner.run_once(im)
        for i,o in enumerate(depth_pyr_conv_optim.optim_output_buffers):
            o_np = o.data()[:17*17].reshape(17,17,1)
            o_np = o_np.repeat(3, axis=-1)
            min_o = np.min(o_np)
            max_o = np.max(o_np)
            num = (o_np-min_o)
            den = (max_o-min_o)
            if den == 0.0:
                num[:]= 0.5
                den = 1.0
            d.imshow(f"im {i}", num/den)
        for i, o in enumerate(depth_pyr_conv.out_pyr_err.get()):
            min_o = np.min(o)
            max_o = np.max(o)
            num = (o - min_o)
            den = (max_o - min_o)
            if den == 0.0:
                num[:] = 0.5
                den = 1.0
            d.imshow(f"im err {i}", num / den)
        pre = depth_pyr_conv.buf_conv_prepool.data().reshape(-1,4, 1)
        pre = pre.repeat(3, axis=-1)
        min_o = np.min(pre)
        max_o = np.max(pre)
        num = (pre - min_o)
        den = (max_o - min_o)
        d.imshow(f"im pre", num/den)
        d.update()
    '''while not d.window.is_closing:
        out_buf = runner.run_once(im)

        for i, o in enumerate(depth_pyr_conv.out_pyr.get()):
            min_o = np.min(o)
            max_o = np.max(o)
            num = (o - min_o)
            den = (max_o - min_o)
            if den == 0.0:
                num[:] = 0.5
                den = 1.0
            d.imshow(f"im {i}", num / den)
        d.update()'''