import kp
import numpy as np
from displayarray import DirectDisplay
import cv2
import os

from sili.core.buffers import ImageBuffer, ImagePyramidBuffer, calculate_pyramid_levels, ConvDepthBuffer, \
    ConvDepthReductionBuffer
from sili.core.devices.gpu import GPUManager, get_shader
from sili.modules.base import Module
file_path = os.path.dirname(os.path.abspath(__file__))


class Adacon(Module):
    def __init__(self,
                 gpu: GPUManager,
                 weight_buffer: kp.Tensor,
                 weight_err_buffer: kp.Tensor,
                 weight_contrib_buffer: kp.Tensor,
                 weight_connection_buffer: kp.Tensor,
                 ):
        """

        :param gpu: the GPUManager
        :param prediction_buffer: the kp.Tensor object
            Backpropagating error is the main reason to use this class, so default true.
        """
        self.gpu = gpu

        if not isinstance(weight_buffer, kp.Tensor):
            raise TypeError("weight_buffer must be a buffer attached to a device")
        else:
            self.w_buf = weight_buffer

        if not isinstance(weight_err_buffer, kp.Tensor):
            raise TypeError("weight_err_buffer must be a buffer attached to a device")
        else:
            self.w_err_buf = weight_err_buffer

        if not isinstance(weight_contrib_buffer, kp.Tensor):
            raise TypeError("weight_contrib_buffer must be a buffer attached to a device")
        else:
            self.w_contrib_buf = weight_contrib_buffer

        if not isinstance(weight_connection_buffer, kp.Tensor):
            raise TypeError("weight_connection_buffer must be a buffer attached to a device")
        else:
            self.w_connect_buf = weight_connection_buffer

        self.has_forward = False
        self.has_backward = False
        self.has_optim = True

        self.optim_shader = get_shader(file_path + os.sep + 'adacon.comp')

        self.optim_input_buffers = [self.w_buf, self.w_err_buf, self.w_contrib_buf, self.w_connect_buf]

        # note: w_contrib and w_err are zeroed here, within the shader
        self.optim_output_buffers = [self.w_buf, self.w_err_buf, self.w_contrib_buf, self.w_connect_buf]

        self.optim_algorithm = self.gpu.manager.algorithm(
            [self.w_buf, self.w_err_buf, self.w_contrib_buf, self.w_connect_buf],
            spirv=self.optim_shader,
            workgroup=[int(np.ceil(self.w_buf.size() / min(self.w_buf.size(), self.gpu.max_workgroup_invocations))), 0,
                       0],
            spec_consts=np.concatenate(
                (np.asarray([int(min(self.gpu.max_workgroup_invocations, self.w_buf.size()))], dtype=np.uint32).view(
                    np.float32), [0.1, 1e-1]))

        )

        self.basic_sequence = None  # mostly for debugging

    def forward_ops(self):
        return []

    def backward_ops(self):
        return []

    def optim_ops(self):
        return [kp.OpAlgoDispatch(self.optim_algorithm)]

    def optim_buffs(self):
        # Doesn't make sense to optimize this. It isn't a neural net.
        return []

    def basic_optim(self):
        # todo: make this a superclass method
        if self.basic_sequence is None:
            self.basic_sequence = self.gpu.manager.sequence()
            self.basic_sequence.record(kp.OpTensorSyncDevice([*self.optim_input_buffers]))
            for o in self.optim_ops():
                self.basic_sequence.record(kp.OpAlgoDispatch(o))
            self.basic_sequence.record(kp.OpTensorSyncLocal([*self.optim_output_buffers]))
        self.basic_sequence.eval()
        return self.w_buf

    def display_basic_optim_sequence(self, shape=None, display=None):
        if display is None:
            display = DirectDisplay()
        err_np = self.basic_optim().data()
        if shape is not None:
            err_np = err_np.reshape(shape)
            min_err = np.min(err_np)
            max_err = np.max(err_np)
            display.imshow(f'output err', (err_np - min_err) / (max_err - min_err))
        else:
            raise ValueError("Cannot display a line that's likely larger than any screen.")
        while True:
            display.update()
            if display.window.is_closing:
                break


if __name__ == '__main__':
    im = cv2.imread("../../../test/files/test_ai_pls_ignore.png").astype(np.float32) / 255
    gpu = GPUManager()
    act_buf = ImageBuffer(gpu, im)
    pred_buf = ImageBuffer(gpu, im + np.random.normal(0.1, 0.5, im.shape))
    err = Adacon(gpu, pred_buf.buffer, act_buf.buffer, forward=False, backprop=True)
    err.display_basic_backward_sequence(shape=im.shape)

    # im_pyr = pyr.run_basic_forward_sequence(im)
    # with open("test_files/test_ai_pyr_pls_ignore.pyr", mode='wb') as f:
    #    pickle.dump(im_pyr, f)
