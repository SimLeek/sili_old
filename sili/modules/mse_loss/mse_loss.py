import kp
import numpy as np
from displayarray import DirectDisplay
import cv2
import os
from sili.modules.base import Module

from sili.core.buffers import ImageBuffer, ImagePyramidBuffer, calculate_pyramid_levels, ConvDepthBuffer, \
    ConvDepthReductionBuffer
from sili.core.devices.gpu import GPUManager, get_shader

file_path = os.path.dirname(os.path.abspath(__file__))


class MSELoss(Module):
    def __init__(self,
                 gpu: GPUManager,
                 prediction_buffer: kp.Tensor,
                 actual_buffer: kp.Tensor,
                 error_buffer: kp.Tensor,
                 forward=False,
                 backprop=True
                 ):
        """ Mean squared error loss.

        :param gpu: the GPUManager
        :param prediction_buffer: the kp.Tensor object
        :param forward: if true, actually calculate the forward error. (it's not necessary)
        :param backprop: if true, backprop error.
            Backpropagating error is the main reason to use this class, so default true.
        """
        self.gpu = gpu
        if not isinstance(prediction_buffer, kp.Tensor):
            raise TypeError("Prediction Buffer must be a buffer attached to a device")
        else:
            self.pred_buff = prediction_buffer

        if not isinstance(actual_buffer, kp.Tensor):
            raise TypeError("Actual Buffer must be a buffer attached to a device")
        else:
            self.act_buff = actual_buffer

        if not isinstance(error_buffer, kp.Tensor):
            raise TypeError("Error Buffer must be a buffer attached to a device")
        else:
            self.err_grad_buff = error_buffer

        self.has_forward = forward
        if forward:
            self.forward_shader = get_shader(file_path + os.sep + 'mse_loss.comp')

            self.err_buff = gpu.buffer(np.zeros_like(self.act_buff.data()))

            # PIPELINE OBJECTS:
            # FORWARD
            # these need to be accessible so that kompute can record input/output for pipelines
            self.forward_input_buffers = [self.pred_buff, self.act_buff]
            self.forward_output_buffers = [self.err_buff]

            # Note: I'm not going to sum the error into one number. It's kinda useful, but it costs a bit and you can do
            # that with a reduction op anywhere. Leaving it like this gives us options for nice error images, while
            # leaving options for making error graphs.
            # Note2: Currently, this reduces to num_workgroups in one step, which is a fair amount of reduction.
            # todo: Either make this return a full image, or do a full reduction to one number.
            self.forward_algorithm = self.gpu.manager.algorithm(
                [self.pred_buff, self.act_buff, self.err_buff],
                spirv=self.forward_shader,
                workgroup=[int(np.ceil(self.pred_buff.size() / self.gpu.max_workgroup_invocations)), 0, 0],
                spec_consts=np.asarray([self.gpu.max_workgroup_invocations], dtype=np.uint32).view(np.float32)
            )

        self.has_backward = backprop
        if backprop:
            self.backward_shader = get_shader(file_path + os.sep + 'mse_loss_backward.comp')

            self.backward_input_buffers = [self.pred_buff, self.act_buff]
            self.backward_output_buffers = [self.err_grad_buff]

            self.backward_algorithm = self.gpu.manager.algorithm(
                [self.pred_buff, self.act_buff, self.err_grad_buff],
                spirv=self.backward_shader,
                workgroup=[int(np.ceil(self.pred_buff.size() / self.gpu.max_workgroup_invocations)), 0, 0],
                spec_consts=np.asarray([self.gpu.max_workgroup_invocations], dtype=np.uint32).view(np.float32)
            )

        self.has_optim = False
        self.basic_sequence = None  # mostly for debugging

    def forward_ops(self):
        if self.has_forward:
            return [
                kp.OpAlgoDispatch(self.forward_algorithm)
            ]
        else:
            return []

    def backward_ops(self):
        if self.has_backward:
            return [
                kp.OpAlgoDispatch(self.backward_algorithm)
            ]
        else:
            return []

    def optim_ops(self):
        return []

    def optim_buffs(self):
        # Doesn't make sense to optimize this. It isn't a neural net.
        return []

    def basic_forward(self):
        # todo: make this a superclass method
        if self.basic_sequence is None:
            self.basic_sequence = self.gpu.manager.sequence()
            self.basic_sequence.record(kp.OpTensorSyncDevice([*self.forward_input_buffers]))
            for f in self.forward_ops():
                self.basic_sequence.record(kp.OpAlgoDispatch(f))
            self.basic_sequence.record(kp.OpTensorSyncLocal([*self.forward_output_buffers]))
        self.basic_sequence.eval()
        return self.err_buff

    def display_basic_forward_sequence(self, shape=None, display=None):
        if display is None:
            display = DirectDisplay()
        err_np = self.basic_forward().data()
        if shape is not None:
            err_np = err_np.reshape(shape)
            min_err = np.min(err_np)
            max_err = np.max(err_np)
            display.imshow(f'output err', (err_np-min_err)/(max_err-min_err))
        else:
            raise ValueError("Cannot display a line that's likely larger than any screen.")
        while True:
            display.update()
            if display.window.is_closing:
                break

    def basic_backward(self):
        # todo: make this a superclass method
        if self.basic_sequence is None:
            self.basic_sequence = self.gpu.manager.sequence()
            self.basic_sequence.record(kp.OpTensorSyncDevice([*self.backward_input_buffers]))
            for b in self.backward_ops():
                self.basic_sequence.record(b)
            self.basic_sequence.record(kp.OpTensorSyncLocal([*self.backward_output_buffers]))
        self.basic_sequence.eval()
        return self.err_grad_buff

    def display_basic_backward_sequence(self, shape=None, display=None):
        if display is None:
            display = DirectDisplay()
        err_np = self.basic_backward().data()
        if shape is not None:
            err_np = err_np.reshape(shape)
            min_err = np.min(err_np)
            max_err = np.max(err_np)
            display.imshow(f'output err', (err_np-min_err)/(max_err-min_err))
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
    pred_buf = ImageBuffer(gpu, im+np.random.normal(0.1, 0.5, im.shape))
    err_buf = ImageBuffer(gpu, np.zeros_like(im))
    err = MSELoss(gpu, pred_buf.buffer, act_buf.buffer, err_buf.buffer, forward=False, backprop=True)
    err.display_basic_backward_sequence(shape=im.shape)

    # im_pyr = pyr.run_basic_forward_sequence(im)
    # with open("test_files/test_ai_pyr_pls_ignore.pyr", mode='wb') as f:
    #    pickle.dump(im_pyr, f)
